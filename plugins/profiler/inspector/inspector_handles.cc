/*************************************************************************
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for license information.
 *************************************************************************/

#include "inspector.h"
#include "inspector_event_pool.h"
#include <condition_variable>
#include <limits>
#include <mutex>
#include <unordered_map>

namespace {
// Low byte: tag bit + 7-bit kind; upper bits: never-reused collective ID.
// No token is dereferenced, including when the registry no longer contains it.
constexpr unsigned TOKEN_SHIFT = 8;
constexpr unsigned PLAN_TOKEN_KIND = 127;
static_assert(PLAN_TOKEN_KIND > MAX_CHANNELS, "plan tokens must never resolve to a collective");
static_assert(sizeof(uintptr_t) == 8, "Inspector tokens require a 64-bit host");
static_assert(alignof(inspectorP2pInfo) >= 2 && alignof(inspectorKernelChInfo) >= 2,
              "P2P pointers must not have the collective token tag");
std::mutex collHandleRegistryMutex;
std::condition_variable collHandleRegistryChanged;
std::unordered_map<uintptr_t, std::shared_ptr<inspectorCollInfo>> collHandleRegistry;
uintptr_t nextCollHandleId = 0; // deliberately not reset by communicator/global finalize

void inspectorReleaseCollObject(inspectorCollInfo* coll) {
  auto* comm = coll->commInfo;
  inspectorPluginCollInfoCleanup(coll);
  std::lock_guard<std::mutex> lock(collHandleRegistryMutex);
  --comm->collHandleObjects;
  collHandleRegistryChanged.notify_all();
}
}

bool inspectorIsCollToken(void* handle) { return (uintptr_t(handle) & 1) != 0; }
unsigned inspectorCollTokenKind(void* handle) { return (uintptr_t(handle) & 255) >> 1; }
void* inspectorCollToken(const inspectorCollInfo* coll, unsigned kind) {
  return reinterpret_cast<void*>((coll->handleId << TOKEN_SHIFT) | (uintptr_t(kind) << 1) | 1);
}

std::shared_ptr<inspectorCollInfo> inspectorPublishCollHandle(inspectorCollInfo* coll) {
  // Establish a counted owner before any allocation can fail. Destruction must
  // occur outside the registry lock: the deleter also updates the communicator count.
  {
    std::lock_guard<std::mutex> lock(collHandleRegistryMutex);
    ++coll->commInfo->collHandleObjects;
  }
  std::shared_ptr<inspectorCollInfo> owner;
  try {
    owner = std::shared_ptr<inspectorCollInfo>(coll, inspectorReleaseCollObject);
    std::lock_guard<std::mutex> lock(collHandleRegistryMutex);
    if (coll->commInfo->collHandlesClosing ||
        nextCollHandleId == (std::numeric_limits<uintptr_t>::max() >> TOKEN_SHIFT)) return {};
    coll->handleId = ++nextCollHandleId;
    collHandleRegistry.emplace(coll->handleId, owner);
  } catch (...) {
    // shared_ptr invokes the deleter itself if allocating its control block fails.
    return {};
  }
  return owner;
}

std::shared_ptr<inspectorCollInfo> inspectorAcquireCollHandle(void* handle) {
  if (!inspectorIsCollToken(handle) || inspectorCollTokenKind(handle) > MAX_CHANNELS) return {};
  std::lock_guard<std::mutex> lock(collHandleRegistryMutex);
  auto it = collHandleRegistry.find(uintptr_t(handle) >> TOKEN_SHIFT);
  if (it == collHandleRegistry.end() || it->second->commInfo->collHandlesClosing) return {};
  return it->second;
}

void inspectorEraseCollHandle(uintptr_t id) {
  std::shared_ptr<inspectorCollInfo> removed;
  {
    std::lock_guard<std::mutex> lock(collHandleRegistryMutex);
    auto it = collHandleRegistry.find(id);
    if (it != collHandleRegistry.end()) {
      removed = std::move(it->second);
      collHandleRegistry.erase(it);
    }
  }
}

void inspectorMaybeReleaseCollHandle(const std::shared_ptr<inspectorCollInfo>& coll) {
  bool release = false;
  inspectorLockWr(&coll->guard);
  if (!coll->retired && coll->refCount == 0 && !coll->activeRegistered) {
    coll->retired = true;
    release = true;
  }
  inspectorUnlockRWLock(&coll->guard);
  if (release) inspectorEraseCollHandle(coll->handleId);
}

void inspectorBeginCloseCollHandles(inspectorCommInfo* comm) {
  std::lock_guard<std::mutex> lock(collHandleRegistryMutex);
  comm->collHandlesClosing = true;
}

void inspectorCloseCollHandles(inspectorCommInfo* comm) {
  // Closing blocks new leases. Extract nodes without destroying their owners
  // under the registry lock. Existing callbacks finish before comm/pool teardown.
  for (;;) {
    std::shared_ptr<inspectorCollInfo> removed;
    {
      std::lock_guard<std::mutex> lock(collHandleRegistryMutex);
      auto it = collHandleRegistry.begin();
      while (it != collHandleRegistry.end() && it->second->commInfo != comm) ++it;
      if (it == collHandleRegistry.end()) break;
      removed = std::move(it->second);
      collHandleRegistry.erase(it);
    }
    inspectorLockWr(&removed->guard);
    removed->retired = true;
    inspectorUnlockRWLock(&removed->guard);
    inspectorUnregisterActiveColl(comm, removed.get());
  }
  std::unique_lock<std::mutex> lock(collHandleRegistryMutex);
  collHandleRegistryChanged.wait(lock, [&] { return comm->collHandleObjects == 0; });
}

template <typename T>
static void inspectorAtomicMax(T* target, T value) {
  T old = __atomic_load_n(target, __ATOMIC_RELAXED);
  while (old < value &&
         !__atomic_compare_exchange_n(target, &old, value,
                                      false, __ATOMIC_RELEASE, __ATOMIC_RELAXED)) {}
}

void inspectorMarkCollStarted(inspectorCollInfo* coll) {
  inspectorAtomicMax(&coll->commInfo->furthestStartedSequence, coll->trackingSequence);
}

void* inspectorPlanToken(uint64_t planSequence) {
  return reinterpret_cast<void*>((uintptr_t(planSequence) << TOKEN_SHIFT) |
                                 (uintptr_t(PLAN_TOKEN_KIND) << 1) | 1);
}
bool inspectorIsPlanToken(void* handle) {
  return inspectorIsCollToken(handle) && inspectorCollTokenKind(handle) == PLAN_TOKEN_KIND;
}
uint64_t inspectorPlanTokenSequence(void* handle) { return uintptr_t(handle) >> TOKEN_SHIFT; }

void inspectorMarkChannelStarted(inspectorCollInfo* coll, unsigned channelId) {
  auto* comm = coll->commInfo;
  inspectorAtomicMax(&comm->observedChannels, uint32_t(channelId + 1));
  if (!coll->trackingSequence) return;
  inspectorAtomicMax(&comm->channelStartedThrough[channelId], coll->trackingSequence);
  if (coll->planSequence) inspectorAtomicMax(&comm->furthestStartedPlan, coll->planSequence);
}

void inspectorMarkChannelCompleted(inspectorCollInfo* coll, unsigned channelId) {
  if (!coll->trackingSequence) return;
  inspectorAtomicMax(&coll->commInfo->channelCompletedThrough[channelId], coll->trackingSequence);
}
