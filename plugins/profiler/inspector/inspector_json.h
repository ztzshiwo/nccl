#ifndef INSPECTOR_INSPECTOR_JSON_H_
#define INSPECTOR_INSPECTOR_JSON_H_

#include "inspector.h"

// JSON-related function declarations
inspectorResult_t inspectorCommInfoListDump(jsonFileOutput* jfo,
                                            struct inspectorCommInfoList* commList);
inspectorResult_t inspectorCollTimeoutDump(
    jsonFileOutput* jfo,
    const std::vector<inspectorCollTimeoutInfo>& timeoutRecords);

#endif  // INSPECTOR_INSPECTOR_JSON_H_
