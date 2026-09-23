// SPDX-License-Identifier: Apache-2.0
#include "inspector.h"
#include "inspector_event_pool.h"
#include <condition_variable>
#include <limits>
#include <mutex>
#include <unordered_map>

namespace {
// Low byte: tag bit + 7-bit kind; upper bits: never-reused collective ID.
// No token is dereferenced, including when the registry no longer contains it.
constexpr unsigned tokenShift = 8;
static_assert(sizeof(uintptr_t) == 8, "Inspector tokens require a 64-bit host");
static_assert(alignof(inspectorP2pInfo) >= 2 && alignof(inspectorKernelChInfo) >= 2,
              "P2P pointers must not have the collective token tag");
std::mutex registryMutex;
std::condition_variable registryChanged;
std::unordered_map<uintptr_t, std::shared_ptr<inspectorCollInfo>> registry;
uintptr_t nextId = 0; // deliberately not reset by communicator/global finalize

void releaseObject(inspectorCollInfo* coll) {
  auto* comm = coll->commInfo;
  inspectorPluginCollInfoCleanup(coll);
  std::lock_guard<std::mutex> lock(registryMutex);
  --comm->collHandleObjects;
  registryChanged.notify_all();
}
}

bool inspectorIsCollToken(void* handle) { return (uintptr_t(handle) & 1) != 0; }
unsigned inspectorCollTokenKind(void* handle) { return (uintptr_t(handle) & 255) >> 1; }
void* inspectorCollToken(const inspectorCollInfo* coll, unsigned kind) {
  return reinterpret_cast<void*>((coll->handleId << tokenShift) | (uintptr_t(kind) << 1) | 1);
}

std::shared_ptr<inspectorCollInfo> inspectorPublishCollHandle(inspectorCollInfo* coll) {
  // Establish a counted owner before any allocation can fail. Destruction must
  // occur outside registryMutex: the deleter also updates the communicator count.
  {
    std::lock_guard<std::mutex> lock(registryMutex);
    ++coll->commInfo->collHandleObjects;
  }
  std::shared_ptr<inspectorCollInfo> owner;
  try {
    owner = std::shared_ptr<inspectorCollInfo>(coll, releaseObject);
    std::lock_guard<std::mutex> lock(registryMutex);
    if (coll->commInfo->collHandlesClosing ||
        nextId == (std::numeric_limits<uintptr_t>::max() >> tokenShift)) return {};
    coll->handleId = ++nextId;
    registry.emplace(coll->handleId, owner);
  } catch (...) {
    // shared_ptr invokes the deleter itself if allocating its control block fails.
    return {};
  }
  return owner;
}

std::shared_ptr<inspectorCollInfo> inspectorAcquireCollHandle(void* handle) {
  if (!inspectorIsCollToken(handle) || inspectorCollTokenKind(handle) > MAX_CHANNELS) return {};
  std::lock_guard<std::mutex> lock(registryMutex);
  auto it = registry.find(uintptr_t(handle) >> tokenShift);
  if (it == registry.end() || it->second->commInfo->collHandlesClosing) return {};
  return it->second;
}

void inspectorEraseCollHandle(uintptr_t id) {
  std::shared_ptr<inspectorCollInfo> removed;
  {
    std::lock_guard<std::mutex> lock(registryMutex);
    auto it = registry.find(id);
    if (it != registry.end()) {
      removed = std::move(it->second);
      registry.erase(it);
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
  std::lock_guard<std::mutex> lock(registryMutex);
  comm->collHandlesClosing = true;
}

void inspectorCloseCollHandles(inspectorCommInfo* comm) {
  // Closing blocks new leases. Extract nodes without destroying their owners
  // under the registry lock. Existing callbacks finish before comm/pool teardown.
  for (;;) {
    std::shared_ptr<inspectorCollInfo> removed;
    {
      std::lock_guard<std::mutex> lock(registryMutex);
      auto it = registry.begin();
      while (it != registry.end() && it->second->commInfo != comm) ++it;
      if (it == registry.end()) break;
      removed = std::move(it->second);
      registry.erase(it);
    }
    inspectorLockWr(&removed->guard);
    removed->retired = true;
    inspectorUnlockRWLock(&removed->guard);
    inspectorUnregisterActiveColl(comm, removed.get());
  }
  std::unique_lock<std::mutex> lock(registryMutex);
  registryChanged.wait(lock, [&] { return comm->collHandleObjects == 0; });
}

void inspectorMarkCollStarted(inspectorCollInfo* coll) {
  auto* sequence = &coll->commInfo->furthestStartedSequence;
  uint64_t old = __atomic_load_n(sequence, __ATOMIC_RELAXED);
  while (old < coll->trackingSequence &&
         !__atomic_compare_exchange_n(sequence, &old, coll->trackingSequence,
                                      false, __ATOMIC_RELEASE, __ATOMIC_RELAXED)) {}
}
