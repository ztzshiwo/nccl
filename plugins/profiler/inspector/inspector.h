/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef INSPECTOR_INSPECTOR_H_
#define INSPECTOR_INSPECTOR_H_

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <memory>

#include "json.h"
#include "common.h"
#include "version.h"

#define MAX_CHANNELS                     64

// Bump when ncclProfiler_t alias changes to a new interface version.
#define NCCL_PROFILER_INTERFACE_VERSION 5

#define INS_CHK(call)                                                   \
  do {                                                                  \
    inspectorResult_t res = call;                                       \
    if (inspectorSuccess != res) {                                      \
      INFO_INSPECTOR("%s:%d -> error %d: %s", __FILE__, __LINE__, res,  \
                     inspectorErrorString(res));                        \
      return res;                                                       \
    }                                                                   \
  } while (0);

#define INS_CHK_GOTO(call, res, label)                                  \
  do {                                                                  \
    res = call;                                                         \
    if (inspectorSuccess != res) {                                      \
      INFO_INSPECTOR("%s:%d -> error %d: %s", __FILE__, __LINE__, res,  \
                     inspectorErrorString(res));                        \
      goto label;                                                       \
    }                                                                   \
  } while (0);

// Lock convenience macros
#define INSPECTOR_LOCK_RD_FLAG(lockRef, lockFlag, debug)        \
  do {                                                          \
    if (!lockFlag) {                                            \
      INS_CHK(inspectorLockRd(lockRef));                        \
    }                                                           \
    lockFlag = true;                                            \
  } while (0);

#define INSPECTOR_LOCK_WR_FLAG(lockRef, lockFlag, debug)        \
  do {                                                          \
    if (!lockFlag) {                                            \
      INS_CHK(inspectorLockWr(lockRef));                        \
    }                                                           \
    lockFlag = true;                                            \
  } while (0);

#define INSPECTOR_UNLOCK_RW_LOCK_FLAG(lockRef, lockFlag, debug) \
  do {                                                          \
    if (lockFlag) {                                             \
      INS_CHK(inspectorUnlockRWLock(lockRef));                  \
    }                                                           \
    lockFlag = false;                                           \
  } while (0);


typedef enum {
  ncclFuncBroadcast = 0,
  ncclFuncReduce = 1,
  ncclFuncAllGather = 2,
  ncclFuncReduceScatter = 3,
  ncclFuncAllReduce = 4,
  ncclFuncSendRecv = 5,
  ncclFuncSend = 6,
  ncclFuncRecv = 7,
  ncclFuncAll2All = 8,
  ncclFuncAllGatherV = 9,
  ncclNumFuncs = 10
} ncclFunc_t;

enum inspectorResult_t : int {
  inspectorSuccess = 0,
  inspectorUninitializedError,
  inspectorMemoryError,
  inspectorFileOpenError,
  inspectorDisabledError,
  inspectorLockError,
  inspectorPthreadError,
  inspectorJsonError,
  inspectorCudaError,
  inspectorBadHash,
  inspectorDeleteUnknownCommError,
  inspectorAddDuplicateCommError,
  inspectorNop,
  inspectorNullTally,
  inspectorGlobalInitError,
  inspectorReturn,
};

typedef enum {
  inspectorTimingSourceKernelGpu = 0,
  inspectorTimingSourceKernelCpu = 1,
  inspectorTimingSourceCollectiveCpu = 2,
} inspectorTimingSource_t;

struct inspectorEventTraceInfo {
  uint64_t ts;
  uint64_t sn;
};

typedef enum {
  NCCL_INSP_EVT_TRK_OP_START = 0,
  NCCL_INSP_EVT_TRK_OP_STOP  = 1,
  NCCL_INSP_EVT_TRK_OP_NEVT  = 2,
} inspectorEventTrkOp_t;

typedef enum {
  NCCL_INSP_EVT_TRK_KERNEL_START = 0,
  NCCL_INSP_EVT_TRK_KERNEL_STOP = 1,
  NCCL_INSP_EVT_TRK_KERNEL_RECORD = 2,
  NCCL_INSP_EVT_TRK_KERNEL_NEVT = 3,
} inspectorEventTrkKernel_t;

struct inspectorEventTrkKernelInfo {
  struct inspectorEventTraceInfo evntTrace[NCCL_INSP_EVT_TRK_KERNEL_NEVT];
};

// Unified event tracking for both collective and P2P operations.
struct inspectorEventTrkOpInfo {
  int sn;
  uint32_t nChannels;
  struct inspectorEventTraceInfo evntTrace[NCCL_INSP_EVT_TRK_OP_NEVT];
  struct inspectorEventTrkKernelInfo kernelCh[MAX_CHANNELS];
};

// Unified record stored in the completed ring buffer for both collective and
// P2P operations.  The isP2p discriminator selects which op-type-specific
// fields (algo/proto vs peer) are valid.
struct inspectorCompletedOpInfo {
  bool isP2p;
  ncclFunc_t func;
  uint64_t sn;
  size_t msgSizeBytes;
  uint32_t nChannels;
  uint64_t execTimeUsecs;
  uint64_t min_execTimeUsecs;
  uint64_t min_channel;
  uint64_t max_channel;
  inspectorTimingSource_t timingSource;
  double algoBwGbs;
  double busBwGbs;
  const char* algo;   // coll only (nullptr for P2P)
  const char* proto;  // coll only (nullptr for P2P)
  int peer;           // P2P only (unused for coll)
  struct inspectorEventTrkOpInfo evtTrk;
};

#include "inspector_ring.h"

enum {
  NCCL_COMM_HASH_LENGTH = 17,
  NCCL_COMM_NAME_MAX = 256
};

struct inspectorCollInfo;
struct inspectorCollTimeoutInfo;

typedef enum {
  inspectorCollTimeoutOperationNotDispatched = 0,
  inspectorCollTimeoutGpuAndCommunicationIssue,
  inspectorCollTimeoutGpuIssue,
  inspectorCollTimeoutCommunicationIssue,
  inspectorCollTimeoutUnknown
} inspectorCollTimeoutReason_t;

enum {
  INSPECTOR_COLL_TIMEOUT_REPORTED_QUEUE = 1 << 0,
  INSPECTOR_COLL_TIMEOUT_REPORTED_EXECUTION = 1 << 1,
};

struct inspectorCommInfo {
  struct inspectorCommInfo* next;

  const char* commName;
  char commNameStr[NCCL_COMM_NAME_MAX];
  uint64_t commHash;
  char commHashStr[NCCL_COMM_HASH_LENGTH];
  int rank;
  int nranks;
  int nnodes;
  int cudaDeviceId;     // CUDA device ID for this communicator
  char deviceUuidStr[37]; // Pre-computed device UUID string for filename generation

  bool dump_coll;
  bool dump_p2p;
  struct inspectorCompletedRing completedCollRing;
  struct inspectorCompletedRing completedP2pRing;
  // FIFO of active collectives in the order Inspector observes them.
  struct inspectorCollInfo* activeHead;
  struct inspectorCollInfo* activeTail;
  struct inspectorCollInfo* scanEnd; // first live node outside the scan window
  uint64_t nextCollSequence;         // protected by guard, across all coll types
  uint64_t furthestStartedSequence;  // atomic builtins; survives completed nodes
  bool collHandlesClosing;          // protected by the handle registry mutex
  size_t collHandleObjects;         // includes callbacks in flight
  uint64_t p2pSeqNum;
  pthread_rwlock_t guard;
};

struct inspectorCollTimeoutInfo {
  char commName[NCCL_COMM_NAME_MAX];
  char commHashStr[NCCL_COMM_HASH_LENGTH];
  int rank;
  int nranks;
  int nnodes;
  ncclFunc_t func;
  uint64_t collSeqNumber;
  char algo[32];
  char proto[32];
  size_t msgSizeBytes;
  uint64_t dumpTimestampUsec;
  uint64_t timeoutElapsedMs;
  bool hasQueueWaitElapsed;
  uint64_t queueWaitElapsedMs;
  bool hasKernelExecutionElapsed;
  uint64_t kernelExecutionElapsedMs;
  uint32_t expectedChannels;
  bool hasEstimatedChannelPayloadBytes;
  uint64_t estimatedChannelPayloadBytes;
  uint32_t startedChannels;
  uint32_t completedChannels;
  uint64_t startedChannelMask;
  uint64_t completedChannelMask;
  inspectorCollTimeoutReason_t reason;
  bool hasValidChannelTiming;
  uint64_t minExecTimeUsecs;
  uint64_t minChannel;
  uint64_t maxExecTimeUsecs;
  uint64_t maxChannel;
  inspectorTimingSource_t timingSource;
};

// Structure to track flush times and file handles per device UUID
struct deviceFlushInfo {
  char deviceUuidStr[37];
  uint64_t lastFlushTime;
  FILE* fileHandle;           // Open file handle for this device
  bool needsCreation;         // Whether file needs to be created/flushed
  char filename[1024];        // Store filename for cleanup
};

struct inspectorDumpThread {
  bool run{false};
  bool threadStarted{false};
  jsonFileOutput* jfo;
  jsonFileOutput* timeoutJfo;
  char* outputRoot;
  char* timeoutOutputRoot;
  int64_t sampleIntervalUsecs;
  uint64_t timeoutCheckIntervalUsecs;
  bool timeoutWriterFailed{false};
  std::vector<deviceFlushInfo> deviceFlushEntries;
  pthread_t pthread;
  pthread_mutex_t stateMutex;
  pthread_cond_t stateCond;

  // Constructor and destructor implemented in inspector.cc where dependencies are available
  inspectorDumpThread(const char* _outputRoot, int64_t _sampleIntervalUsecs);
  ~inspectorDumpThread();

  /*
   * Gets or creates a file handle for a device UUID, handling flushing as needed.
   */
  FILE* getOrCreateFileHandle(const char* deviceUuidStr,
                              const char* filename,
                              uint64_t currentTime);

  void startThread();
  void stopThread();
  inspectorResult_t inspectorStateDump(const char* output_root);
  inspectorResult_t inspectorStateDumpJSON(const char* output_root);
  inspectorResult_t inspectorStateDumpProm(const char* output_root);
  inspectorResult_t inspectorTimeoutDump(
      const std::vector<inspectorCollTimeoutInfo>& timeoutRecords);
  static void* dumpMain(void* arg);
};

struct inspectorKernelChInfo {
  uint64_t type;
  int refCount; /*unused*/
  uint64_t parentType;  // ncclProfileColl or ncclProfileP2p
  void* parentObj;      // Pointer to either inspectorCollInfo or inspectorP2pInfo
  uint8_t channelId;
  uint64_t tsStartUsec;
  uint64_t tsCompletedUsec;
  uint64_t startGpuClk;
  uint64_t stopGpuClk;
};

struct inspectorCollInfo {
  uint64_t type;
  int refCount; // outstanding normal callbacks; registry/leases own the storage
  struct inspectorCommInfo *commInfo;
  const char* func;
  const char* algo;
  const char* proto;
  uint64_t sn;
  size_t msgSizeBytes;
  uint64_t tsStartUsec;
  uint64_t tsCompletedUsec;
  uint32_t nChannels;
  uint32_t nKernelChStarted;
  uint32_t nKernelChCompleted;
  uint64_t startedChannelMask;
  uint64_t recordRefReleasedChannelMask;
  uint64_t completedChannelMask;
  bool kernelCompletionRefHeld;
  // Monotonic timestamp when this collective became FIFO-head eligible.
  // Retained after execution starts so timeout JSON can report queue wait.
  uint64_t queueWatchStartUsec;
  // Monotonic timestamp of the first KernelCh start callback.
  uint64_t executionStartUsec;
  uint8_t timeoutReportedMask;
  bool activeRegistered;
  bool retired;
  bool parentStopped;
  uintptr_t handleId;
  uint64_t trackingSequence;
  struct inspectorCollInfo* activePrev;
  struct inspectorCollInfo* activeNext;
  pthread_rwlock_t guard;
  struct inspectorKernelChInfo kernelCh[MAX_CHANNELS];
  struct inspectorEventTrkOpInfo collEvtTrk;
};

struct inspectorP2pInfo {
  uint64_t type;
  int refCount;
  struct inspectorCommInfo *commInfo;
  const char* func;
  uint64_t sn;
  size_t msgSizeBytes;
  uint64_t tsStartUsec;
  uint64_t tsCompletedUsec;
  uint32_t nChannels;
  uint32_t nKernelChStarted;
  uint32_t nKernelChCompleted;
  pthread_rwlock_t guard;
  struct inspectorKernelChInfo kernelCh[MAX_CHANNELS];
  struct inspectorEventTrkOpInfo p2pEvtTrk;
  int peer;
};

struct inspectorCommInfoList {
  struct inspectorCommInfo* comms;
  uint32_t ncomms;
  pthread_rwlock_t guard;
};

struct inspectorState {
  struct inspectorCommInfoList liveComms;
  struct inspectorCommInfoList deletedComms;
};



extern ncclDebugLogger_t logFn;
#define VERSION(...) logFn(NCCL_LOG_VERSION, NCCL_ALL, __FILE__, __LINE__, __VA_ARGS__)
#define INFO(FLAGS, ...) logFn(NCCL_LOG_INFO, (FLAGS), __func__, __LINE__, __VA_ARGS__)

// Use NCCL_PROFILE for inspector messages so they can be filtered with NCCL_DEBUG_SUBSYS=PROFILE
#define INFO_INSPECTOR(...) logFn(NCCL_LOG_INFO, NCCL_PROFILE, __func__, __LINE__, __VA_ARGS__)
#define TRACE_INSPECTOR(...) logFn(NCCL_LOG_TRACE, NCCL_PROFILE, __func__, __LINE__, __VA_ARGS__)
// Set NCCL_INSPECTOR_ENABLE_WARN=1 to enable WARN-level inspector logging (default: 0 -> INFO)
#if NCCL_INSPECTOR_ENABLE_WARN
#define WARN_INSPECTOR(...) logFn(NCCL_LOG_WARN, NCCL_PROFILE, __func__, __LINE__, __VA_ARGS__)
#else
#define WARN_INSPECTOR(...) INFO_INSPECTOR(__VA_ARGS__)
#endif

inline int ncclTypeSize(ncclDataType_t type) {
  switch (type) {
  case ncclInt8:
  case ncclUint8:
  case ncclFloat8e4m3:
  case ncclFloat8e5m2:
    return 1;
  case ncclFloat16:
  case ncclBfloat16:
    return 2;
  case ncclInt32:
  case ncclUint32:
  case ncclFloat32:
    return 4;
  case ncclInt64:
  case ncclUint64:
  case ncclFloat64:
    return 8;
  default:
    return -1;
  }
}

// Global flag to control P2P tracking
extern bool enableNcclInspectorP2p;
extern bool requireKernelTiming;
// Minimum message size (bytes) to be `tracked by inspector
extern size_t ncclInspectorDumpMinSizeBytes;

bool inspectorIsDumpVerboseEnabled();

const char* inspectorErrorString(inspectorResult_t result);

inspectorResult_t inspectorLockInit(pthread_rwlock_t* lockRef);
inspectorResult_t inspectorLockDestroy(pthread_rwlock_t* lockRef);
inspectorResult_t inspectorLockRd(pthread_rwlock_t* lockRef);
inspectorResult_t inspectorLockWr(pthread_rwlock_t* lockRef);
inspectorResult_t inspectorUnlockRWLock(pthread_rwlock_t* lockRef);
inspectorResult_t inspectorGlobalInit(int rank);
inspectorResult_t inspectorGlobalFinalize();
uint64_t inspectorGetTime();
uint64_t inspectorGetMonotonicTimeUsecs();
bool inspectorCollTimeoutEnabled();
void inspectorRegisterActiveColl(struct inspectorCommInfo* commInfo,
                                 struct inspectorCollInfo* collInfo);
bool inspectorUnregisterActiveColl(struct inspectorCommInfo* commInfo,
                                   struct inspectorCollInfo* collInfo);
inspectorResult_t inspectorPluginCollInfoDeRef(struct inspectorCollInfo* collInfo);
void inspectorPluginCollInfoCleanup(struct inspectorCollInfo* collInfo);
// Collective handles are opaque tagged IDs. Registry leases, not callback
// expectations, own the memory. P2P handles remain aligned object pointers.
bool inspectorIsCollToken(void* handle);
unsigned inspectorCollTokenKind(void* handle); // 0 = parent, channel + 1 = child
void* inspectorCollToken(const inspectorCollInfo* coll, unsigned kind = 0);
std::shared_ptr<inspectorCollInfo> inspectorPublishCollHandle(inspectorCollInfo* coll);
std::shared_ptr<inspectorCollInfo> inspectorAcquireCollHandle(void* handle);
void inspectorEraseCollHandle(uintptr_t id);
void inspectorMaybeReleaseCollHandle(const std::shared_ptr<inspectorCollInfo>& coll);
void inspectorBeginCloseCollHandles(inspectorCommInfo* comm);
void inspectorCloseCollHandles(inspectorCommInfo* comm);
void inspectorMarkCollStarted(inspectorCollInfo* coll);
inspectorResult_t inspectorGetTimeUTC(char* buffer, size_t bufferSize);
inspectorResult_t inspectorAddComm(struct inspectorCommInfo **commInfo,
                                   const char* commName, uint64_t commHash,
                                   int nNodes, int nranks, int rank);
inspectorResult_t inspectorDelComm(struct inspectorCommInfo *commInfo);

void inspectorUpdateCollPerf(struct inspectorCompletedOpInfo *completedOp,
                             struct inspectorCollInfo *collInfo);
void inspectorUpdateP2pPerf(struct inspectorCompletedOpInfo *completedOp,
                            struct inspectorP2pInfo *p2pInfo);
ncclDataType_t inspectorStringToDatatype(const char* str);

void inspectorComputeOpBw(struct inspectorCommInfo *commInfo,
                          struct inspectorCompletedOpInfo *op);

// Utility functions exposed for Prometheus module
const char* inspectorTimingSourceToString(inspectorTimingSource_t timingSource);
inspectorResult_t inspectorCommInfoListFinalize(struct inspectorCommInfoList* commList);
const char* ncclFuncToString(ncclFunc_t fn);
ncclFunc_t ncclStringToFunc(const char* str);

// Global state
extern struct inspectorState g_state;

#endif  // INSPECTOR_INSPECTOR_H_
