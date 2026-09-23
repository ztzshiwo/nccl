// CPU-only integration: real Inspector callbacks/scanner, controllable clock,
// host CUDA declarations. No CUDA/NCCL communication or timing claims.
#include "../inspector.cc"
#include <atomic>
#include <cassert>
#include <chrono>
#include <future>
#include <iostream>
#include <thread>

extern ncclProfiler_t ncclProfiler_v5;
extern "C" const char* get_git_version_info() { return "cpu-lifecycle-test"; }
static void logger(ncclDebugLogLevel, unsigned long, const char*, int, const char*, ...) {}
static uint64_t commSerial = 0;
static jsonFileOutput* regressionJson = nullptr;
static void now(uint64_t t) { __atomic_store_n(&inspectorTestNowUsec, t, __ATOMIC_RELAXED); }
static size_t poolCount() {
  pthread_mutex_lock(&g_eventPool.collPoolLock);
  size_t n = g_eventPool.collAllocCount;
  pthread_mutex_unlock(&g_eventPool.collPoolLock);
  return n;
}

struct Fixture {
  inspectorCommInfo* comm;
  Fixture() {
    now(1000000);
    ncclInspectorCollTimeoutUsecs = 30000000;
    ncclInspectorCollTimeoutOutputEnabled = true;
    comm = static_cast<inspectorCommInfo*>(calloc(1, sizeof(inspectorCommInfo)));
    comm->commName = "cpu-test"; comm->commHash = ++commSerial;
    comm->nranks = 2; comm->nnodes = 1;
    snprintf(comm->commHashStr, sizeof(comm->commHashStr), "%llu", (unsigned long long)commSerial);
    assert(inspectorLockInit(&comm->guard) == inspectorSuccess);
    assert(inspectorRingInit(&comm->completedCollRing, 8, sizeof(inspectorCompletedOpInfo)) == inspectorSuccess);
    assert(inspectorRingInit(&comm->completedP2pRing, 8, sizeof(inspectorCompletedOpInfo)) == inspectorSuccess);
    inspectorLockWr(&g_state.liveComms.guard);
    comm->next = g_state.liveComms.comms;
    g_state.liveComms.comms = comm; ++g_state.liveComms.ncomms;
    inspectorUnlockRWLock(&g_state.liveComms.guard);
  }
  ~Fixture() {
    if (comm) assert(inspectorDelComm(comm) == inspectorSuccess);
    assert(inspectorCommInfoListFinalize(&g_state.deletedComms) == inspectorSuccess);
    assert(poolCount() == 0);
  }
  void* coll(int channels, const char* func = "AllReduce") {
    ncclProfilerEventDescr_t d = {};
    d.type = ncclProfileColl; d.coll.func = func; d.coll.datatype = "ncclFloat32";
    d.coll.count = 32768; d.coll.nChannels = channels;
    d.coll.algo = "RING"; d.coll.proto = "SIMPLE";
    void* h = nullptr;
    assert(ncclProfiler_v5.startEvent(comm, &h, &d) == ncclSuccess);
    assert(h && inspectorIsCollToken(h));
    return h;
  }
  void* start(void* parent, int channel) {
    ncclProfilerEventDescr_t d = {};
    d.type = ncclProfileKernelCh; d.parentObj = parent;
    d.kernelCh.channelId = channel; d.kernelCh.pTimer = 1000000;
    void* h = nullptr;
    assert(ncclProfiler_v5.startEvent(comm, &h, &d) == ncclSuccess);
    return h;
  }
  void complete(void* h) {
    record(h); stop(h);
  }
  static void stop(void* h) { assert(ncclProfiler_v5.stopEvent(h) == ncclSuccess); }
  static void record(void* h) {
    ncclProfilerEventStateArgs_t args = {}; args.kernelCh.pTimer = 1100000;
    assert(ncclProfiler_v5.recordEventState(h, ncclProfilerKernelChStop, &args) == ncclSuccess);
  }
  std::vector<inspectorCollTimeoutInfo> scan() {
    std::vector<inspectorCollTimeoutInfo> out;
    assert(inspectorCollectTimedOutColls(out) == inspectorSuccess);
    return out;
  }
};

static void thresholdsAndReuse() {
  Fixture f;
  auto a = f.coll(24);
  auto owner = inspectorAcquireCollHandle(a);
  auto* oldAddress = owner.get();
  void* oldChild = nullptr;
  for (int i = 0; i < 23; ++i) { oldChild = f.start(a, i); f.complete(oldChild); }
  f.stop(a);
  now(30999000); assert(f.scan().empty()); assert(!owner->retired);
  now(31000000); assert(f.scan().empty()); assert(owner->retired);
  assert(!f.comm->activeHead && !inspectorAcquireCollHandle(a));
  assert(poolCount() == 1); // in-flight owner lease keeps the allocation alive
  owner.reset(); assert(poolCount() == 0);
  auto b = f.coll(2);
  auto replacement = inspectorAcquireCollHandle(b);
  assert(replacement.get() == oldAddress && a != b); // deterministic LIFO pool reuse
  assert(!f.start(a, 23)); f.record(oldChild); f.stop(oldChild); f.stop(a);
  assert(replacement->nKernelChStarted == 0 && replacement->refCount == 2);
  replacement.reset();
  uint64_t visits = inspectorTestScanVisits;
  for (int i = 0; i < 10; ++i) assert(f.scan().empty());
  assert(inspectorTestScanVisits - visits == 10); // only B, never retired A
  f.complete(f.start(b, 0)); f.complete(f.start(b, 1)); f.stop(b);
  assert(poolCount() == 0 && !f.comm->activeHead && !f.comm->scanEnd);
  assert(inspectorRingNonEmpty(&f.comm->completedCollRing));
  // Queue-only 1/0/0 follows the configured stage budget, not a hardcoded 30s.
  ncclInspectorCollTimeoutUsecs = 7000000;
  auto c = f.coll(1);
  now(37999999); assert(f.scan().empty()); assert(inspectorAcquireCollHandle(c));
  now(38000000); assert(f.scan().empty()); assert(!inspectorAcquireCollHandle(c));
  assert(poolCount() == 0);
  assert(inspectorCommInfoListDump(regressionJson, &g_state.liveComms) == inspectorSuccess);
}

static void frontierAndGaps() {
  Fixture f;
  auto a = f.coll(2); f.complete(f.start(a, 0)); f.stop(a);
  auto b = f.coll(2, "AllGather");
  auto c = f.coll(1, "ReduceScatter");
  auto pending = f.coll(1);
  now(5000000); f.complete(f.start(c, 0)); f.stop(c);
  assert(!inspectorAcquireCollHandle(c));
  now(6000000); assert(f.scan().empty());
  auto bo = inspectorAcquireCollHandle(b);
  auto po = inspectorAcquireCollHandle(pending);
  assert(bo->queueWatchStartUsec == 6000000); // C completed before scan; watermark survives
  assert(po->queueWatchStartUsec == 0 && f.comm->scanEnd == po.get());
  now(31000000); assert(f.scan().empty()); assert(!inspectorAcquireCollHandle(a));
  assert(f.comm->activeHead == bo.get() && bo->queueWatchStartUsec == 6000000);
  now(36000000); auto out = f.scan();
  assert(out.size() == 1 && out[0].reason == inspectorCollTimeoutOperationNotDispatched);
  // B can start after its queue timeout; its execution budget starts afresh.
  f.complete(f.start(b, 0));
  assert(bo->executionStartUsec == 36000000);
  now(65000000); assert(f.scan().empty());
  now(66000000); assert(f.scan().empty()); assert(bo->retired);
  assert(f.comm->activeHead == po.get() && po->queueWatchStartUsec == 66000000);
  bo.reset(); po.reset();
  now(96000000); assert(f.scan().empty()); assert(poolCount() == 0);
  // Empty-window reinsertion and deletion of a pending tail.
  auto d = f.coll(1); auto e = f.coll(1);
  f.complete(f.start(e, 0)); f.stop(e);
  assert(f.comm->activeTail == inspectorAcquireCollHandle(d).get());
  f.complete(f.start(d, 0)); f.stop(d);
  assert(!f.comm->activeTail && !f.comm->scanEnd && poolCount() == 0);
}

static void nonExemptAndCallbacks() {
  for (int mode = 0; mode != 4; ++mode) {
    Fixture f;
    auto h = f.coll(4);
    int starts = mode == 0 ? 2 : mode == 1 ? 4 : 3;
    for (int i = 0; i < starts; ++i) {
      auto ch = f.start(h, i);
      if (mode == 0 || mode == 3 || i < starts - 1) f.complete(ch);
    }
    auto owner = inspectorAcquireCollHandle(h);
    if (mode == 3) owner->completedChannelMask = 11; // same count, different IDs
    now(31000000); auto out = f.scan();
    assert(out.size() == 1 && !owner->retired);
    assert(out[0].reason == (mode == 1 ? inspectorCollTimeoutCommunicationIssue :
                           mode == 2 ? inspectorCollTimeoutGpuAndCommunicationIssue :
                                       inspectorCollTimeoutGpuIssue));
    assert(inspectorCollTimeoutDump(regressionJson, out) == inspectorSuccess);
    assert(f.scan().empty());
  }
  Fixture f;
  auto h = f.coll(1);
  auto ch = f.start(h, 63); // nonzero/high channel ID, count is not an ID range
  f.stop(h); f.stop(ch);
  assert(inspectorAcquireCollHandle(h)); // record callback still outstanding
  f.record(ch); assert(!inspectorAcquireCollHandle(h));
  f.record(ch); f.stop(ch); // old duplicate callbacks remain harmless
  assert(poolCount() == 0);
  ncclInspectorCollTimeoutOutputEnabled = false;
  auto off = f.coll(1); assert(!f.comm->activeHead);
  f.complete(f.start(off, 0)); f.stop(off); assert(poolCount() == 0);
  // P2P remains an aligned pointer and takes the existing callback path.
  ncclProfilerEventDescr_t d = {};
  d.type = ncclProfileP2p; d.p2p.func = "Send"; d.p2p.datatype = "ncclFloat32";
  d.p2p.count = 32768; d.p2p.nChannels = 1;
  void* p = nullptr; assert(ncclProfiler_v5.startEvent(f.comm, &p, &d) == ncclSuccess);
  assert(p && !inspectorIsCollToken(p));
  auto pc = f.start(p, 0); assert(pc && !inspectorIsCollToken(pc));
  f.complete(pc); f.stop(p);
  assert(g_eventPool.p2pAllocCount == 0 && inspectorRingNonEmpty(&f.comm->completedP2pRing));
  assert(inspectorCommInfoListDump(regressionJson, &g_state.liveComms) == inspectorSuccess);
}

static void finalizeWaitsForLease() {
  Fixture f;
  auto h = f.coll(2); f.complete(f.start(h, 0)); f.start(h, 1);
  auto lease = inspectorAcquireCollHandle(h);
  std::promise<void> entering;
  auto closing = std::async(std::launch::async, [&] {
    entering.set_value(); return inspectorDelComm(f.comm);
  });
  entering.get_future().wait();
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (inspectorAcquireCollHandle(h)) {
    assert(std::chrono::steady_clock::now() < deadline);
    std::this_thread::yield();
  }
  assert(closing.wait_for(std::chrono::milliseconds(20)) == std::future_status::timeout);
  lease.reset(); assert(closing.get() == inspectorSuccess);
  f.comm = nullptr;
  assert(!inspectorAcquireCollHandle(h) && poolCount() == 0);
}
static void concurrentProducers() {
  Fixture f;
  std::atomic<int> finished{0};
  std::vector<std::thread> producers;
  for (int t = 0; t < 4; ++t) producers.emplace_back([&] {
    for (int i = 0; i < 250; ++i) {
      auto h = f.coll(3); f.stop(h);
      f.complete(f.start(h, 7)); f.complete(f.start(h, 31));
    }
    ++finished;
  });
  while (finished.load() != 4) { assert(f.scan().empty()); std::this_thread::yield(); }
  for (auto& t : producers) t.join();
  assert(poolCount() == 1000);
  now(31000000); assert(f.scan().empty());
  assert(poolCount() == 0 && !f.comm->activeHead && !f.comm->scanEnd);
}

static void callbackRaces() {
  for (int i = 0; i < 200; ++i) {
    Fixture f;
    auto h = f.coll(2); f.complete(f.start(h, 0)); f.stop(h);
    now(31000000);
    std::atomic<bool> go{false};
    std::thread callback([&] {
      while (!go.load(std::memory_order_acquire)) std::this_thread::yield();
      auto ch = f.start(h, 1);
      if (ch) f.complete(ch);
    });
    go.store(true, std::memory_order_release);
    auto out = f.scan(); // may observe missing-one, in-flight, or fully completed
    callback.join();
    assert(poolCount() == 0 && !f.comm->activeHead);
  }
}

static void cpuCosts() {
  Fixture f;
  std::vector<void*> handles;
  for (int i = 0; i < 4096; ++i) handles.push_back(f.coll(2));
  auto owner = inspectorAcquireCollHandle(handles[0]);
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < 100000; ++i) { inspectorLockWr(&owner->guard); inspectorUnlockRWLock(&owner->guard); }
  auto middle = std::chrono::steady_clock::now();
  for (int i = 0; i < 100000; ++i) {
    auto lease = inspectorAcquireCollHandle(handles[0]);
    inspectorLockWr(&lease->guard); inspectorUnlockRWLock(&lease->guard);
  }
  auto end = std::chrono::steady_clock::now();
  auto scanStart = std::chrono::steady_clock::now();
  uint64_t baselineVisits = 0;
  for (int i = 0; i < 100; ++i) {
    inspectorLockWr(&f.comm->guard);
    for (auto* c = f.comm->activeHead; c; c = c->activeNext) {
      inspectorLockWr(&c->guard); ++baselineVisits; inspectorUnlockRWLock(&c->guard);
    }
    inspectorUnlockRWLock(&f.comm->guard);
  }
  auto scanMiddle = std::chrono::steady_clock::now();
  auto visits = inspectorTestScanVisits;
  for (int i = 0; i < 100; ++i) assert(f.scan().empty());
  auto scanEnd = std::chrono::steady_clock::now();
  assert(inspectorTestScanVisits - visits == 100 && baselineVisits == 409600);
  auto ns = [](auto d) { return std::chrono::duration<double, std::nano>(d).count() / 100000; };
  std::cout << "CPU baseline_object_lock_ns=" << ns(middle-start)
            << " registry_lease_plus_lock_ns=" << ns(end-middle)
            << " scan_visits_per_pass=1 pending_records=4096"
            << " full_list_lock_walk_us=" << std::chrono::duration<double, std::micro>(scanMiddle-scanStart).count()/100
            << " window_scan_us=" << std::chrono::duration<double, std::micro>(scanEnd-scanMiddle).count()/100 << "\n";
}

int main(int argc, char** argv) {
  assert(argc == 2);
  assert(jsonInitFileOutput(&regressionJson, argv[1]) == jsonSuccess);
  logFn = logger; ncclInspectorDumpMinSizeBytes = 0; requireKernelTiming = false;
  assert(inspectorGlobalStateInit() == inspectorSuccess);
  assert(inspectorEventPoolInit(16, 16, 1) == inspectorSuccess);
  thresholdsAndReuse(); frontierAndGaps(); nonExemptAndCallbacks();
  finalizeWaitsForLease(); callbackRaces(); concurrentProducers(); cpuCosts();
  assert(inspectorEventPoolFinalize() == inspectorSuccess);
  assert(jsonFinalizeFileOutput(regressionJson) == jsonSuccess);
  inspectorLockDestroy(&g_state.liveComms.guard); inspectorLockDestroy(&g_state.deletedComms.guard);
  std::cout << "PASS threshold, discard, frontier, gaps, reuse, late callbacks, P2P, finalize, races\n";
}
