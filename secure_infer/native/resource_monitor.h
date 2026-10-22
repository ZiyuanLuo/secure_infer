#pragma once

#include "secure_infer.h"
#include "hack.h"
#include <cstdlib>
#include <cstdio>

static inline bool resource_monitor_enabled() {
    static bool checked = false;
    static bool enabled = true;
    if (!checked) {
        auto e = std::getenv("PIPE_ENABLE_RESOURCE_MONITOR");
        if (e && std::atoi(e) == 0) enabled = false;
        checked = true;
    }
    return enabled;
}

struct IdleThresholds {
    int cube;
    int vector;
    int aicpu;
    int memory;
};

static inline IdleThresholds get_idle_thresholds() {
    IdleThresholds t;
    auto e = std::getenv("PIPE_IDLE_CUBE_THRESHOLD");
    t.cube = e ? std::atoi(e) : 30;
    e = std::getenv("PIPE_IDLE_VECTOR_THRESHOLD");
    t.vector = e ? std::atoi(e) : 30;
    e = std::getenv("PIPE_IDLE_AICPU_THRESHOLD");
    t.aicpu = e ? std::atoi(e) : 30;
    e = std::getenv("PIPE_IDLE_MEMORY_THRESHOLD");
    t.memory = e ? std::atoi(e) : 50;
    return t;
}

static inline bool query_device_utilization(int32_t deviceId, aclrtUtilizationInfo *util) {
    util->utilizationExtend = nullptr;
    aclError ret = real_aclrtGetDeviceUtilizationRate(deviceId, util);
    if (ret != ACL_SUCCESS) {
        fprintf(stderr, "[ResourceMonitor] aclrtGetDeviceUtilizationRate failed: %d\n", ret);
        fflush(stderr);
        return false;
    }
    return true;
}

static inline bool is_ai_core_idle(int32_t deviceId) {
    if (!resource_monitor_enabled()) return true;
    static IdleThresholds thresh = get_idle_thresholds();
    aclrtUtilizationInfo util;
    if (!query_device_utilization(deviceId, &util)) {
        fprintf(stderr, "[ResourceMonitor] is_ai_core_idle: query failed, assuming idle\n");
        fflush(stderr);
        return true;
    }
    bool idle = (util.cubeUtilization < thresh.cube) &&
                (util.vectorUtilization < thresh.vector);
    if (!idle) {
        fprintf(stderr, "[ResourceMonitor] AI Core BUSY: cube=%d%%, vector=%d%% (thresholds: cube<%d%%, vector<%d%%)\n",
                util.cubeUtilization, util.vectorUtilization, thresh.cube, thresh.vector);
        fflush(stderr);
    }
    return idle;
}

static inline bool is_ai_cpu_idle(int32_t deviceId) {
    if (!resource_monitor_enabled()) return true;
    static IdleThresholds thresh = get_idle_thresholds();
    aclrtUtilizationInfo util;
    if (!query_device_utilization(deviceId, &util)) {
        fprintf(stderr, "[ResourceMonitor] is_ai_cpu_idle: query failed, assuming idle\n");
        fflush(stderr);
        return true;
    }
    bool idle = (util.aicpuUtilization < thresh.aicpu);
    if (!idle) {
        fprintf(stderr, "[ResourceMonitor] AI CPU BUSY: aicpu=%d%% (threshold: <%d%%)\n",
                util.aicpuUtilization, thresh.aicpu);
        fflush(stderr);
    }
    return idle;
}

static inline void log_device_utilization(int32_t deviceId) {
    if (!resource_monitor_enabled()) return;
    aclrtUtilizationInfo util;
    if (!query_device_utilization(deviceId, &util)) {
        fprintf(stderr, "[ResourceMonitor] log_device_utilization: query failed\n");
        fflush(stderr);
        return;
    }
    fprintf(stderr, "[ResourceMonitor] Device %d util: cube=%d%%, vector=%d%%, aicpu=%d%%, memory=%d%%\n",
            deviceId, util.cubeUtilization, util.vectorUtilization,
            util.aicpuUtilization, util.memoryUtilization);
    fflush(stderr);
}
