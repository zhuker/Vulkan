#include "parseproc.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <sys/fcntl.h>

#define MAX_MEMINFO_KEY_SIZE  16

bool RBX::MemoryStats::parseMemoryInfoBuf(char* buf, ProcMemInfo& meminfo)
{
    char* head = buf;
    char* tail = nullptr;
    for (;;)
    {
        tail = strchr(head, ':');
        if (!tail)
            break;
        *tail = '\0';
        if (strlen(head) >= MAX_MEMINFO_KEY_SIZE)
        {
            head = tail + 1;
            tail = strchr(head, '\n');
            if (!tail)
                break;
            head = tail + 1;
            continue;
        }
        if (!strncmp("MemTotal", head, MAX_MEMINFO_KEY_SIZE))
        {
            head = tail + 1;
            meminfo.total = strtoul(head, nullptr, 10);
        }
        else if (!strncmp("MemFree", head, MAX_MEMINFO_KEY_SIZE))
        {
            head = tail + 1;
            meminfo.free = strtoul(head, nullptr, 10);
        }
        else if (!strncmp("Cached", head, MAX_MEMINFO_KEY_SIZE))
        {
            head = tail + 1;
            meminfo.cached = strtoul(head, nullptr, 10);
        }
        else if (!strncmp("Buffers", head, MAX_MEMINFO_KEY_SIZE))
        {
            head = tail + 1;
            meminfo.buffers = strtoul(head, nullptr, 10);
        }
        else if (!strncmp("MemAvailable", head, MAX_MEMINFO_KEY_SIZE))
        {
            head = tail + 1;
            meminfo.memAvailable = strtoul(head, nullptr, 10);
        }
        else if (!strncmp("SwapTotal", head, MAX_MEMINFO_KEY_SIZE))
        {
            head = tail + 1;
            meminfo.swapTotal = strtoul(head, nullptr, 10);
        }
        else if (!strncmp("SwapFree", head, MAX_MEMINFO_KEY_SIZE))
        {
            head = tail + 1;
            meminfo.swapFree = strtoul(head, nullptr, 10);
        }
        else
        {
            head = tail + 1;
        }
        tail = strchr(head, '\n');
        if (!tail)
            break;
        head = tail + 1;
    }
    return true;
}

bool RBX::MemoryStats::parseMemoryInfo(ProcMemInfo& meminfo)
{
    char buf[4096];

    // const char* pathname = "../meminfo.txt";
    const char* pathname = "/proc/meminfo";
    int fd = open(pathname, O_RDONLY);
    if (fd < 0)
        return false;
    auto r = read(fd, buf, sizeof(buf));
    if (r <= 0)
    {
        close(fd);
        return false;
    }
    buf[r] = '\0';
    close(fd);
    parseMemoryInfoBuf(buf, meminfo);
    return true;
}

bool RBX::MemoryStats::parseVmstat(const char* buf, Vmstat& vmstat)
{
    const char* start = buf;

    while (true)
    {
        auto tab = strchr(start, ' ');
        if (!tab) break;
        auto eol = strchr(tab, '\n');
        if (!eol)
        {
            break;
        }
        if (!strncmp(start, "pswpin", sizeof("pswpin") - 1))
        {
            vmstat.pswpin = strtoul(tab, nullptr, 10);
        }
        else if (!strncmp(start, "pswpout", sizeof("pswpout") - 1))
        {
            vmstat.pswpout = strtoul(tab, nullptr, 10);
        }
        eol = strchr(start, '\n');
        if (!eol) break;
        start = eol + 1;
    }
    return true;
}

bool RBX::MemoryStats::parsePidStatus(const char* buf, ProcPidStatus& status)
{
    const char* start = buf;

    while (true)
    {
        auto tab = strchr(start, '\t');
        if (!tab) break;
        auto eol = strchr(tab, '\n');
        if (!eol)
        {
            break;
        }
        if (!strncmp(start, "VmSwap:", sizeof("VmSwap:") - 1))
        {
            status.vmSwap = strtoull(tab, nullptr, 10);
        }
        else if (!strncmp(start, "VmSize:", sizeof("VmSize:") - 1))
        {
            status.vmSize = strtoull(tab, nullptr, 10);
        }
        else if (!strncmp(start, "VmRSS:", sizeof("VmRSS:") - 1))
        {
            status.vmRss = strtoull(tab, nullptr, 10);
        }
        else if (!strncmp(start, "RssFile:", sizeof("RssFile:") - 1))
        {
            status.rssFile = strtoull(tab, nullptr, 10);
        }
        eol = strchr(start, '\n');
        if (!eol) break;
        start = eol + 1;
    }
    return true;
}

bool RBX::MemoryStats::parsePidStat(const char* procstat, ProcPidStat& ps)
{
    int rv = sscanf(procstat,
                    "%d %s %c %d %d %d %d %d %u %lu %lu %lu %lu %lu %lu %ld %ld %ld %ld %ld %ld %llu %lu %ld %lu %lu %lu %lu %lu %lu %lu %lu %lu %lu %lu %lu %lu %d %d %u %u %llu %lu %ld %lu %lu %lu %lu %lu %lu %lu %d",
                    &ps.pid,
                    &ps.comm,
                    &ps.state,
                    &ps.ppid,
                    &ps.pgrp,
                    &ps.session,
                    &ps.tty_nr,
                    &ps.tpgid,
                    &ps.flags,
                    &ps.minflt,
                    &ps.cminflt,
                    &ps.majflt,
                    &ps.cmajflt,
                    &ps.utime,
                    &ps.stime,
                    &ps.cutime,
                    &ps.cstime,
                    &ps.priority,
                    &ps.nice,
                    &ps.num_threads,
                    &ps.itrealvalue,
                    &ps.starttime,
                    &ps.vsize,
                    &ps.rss,
                    &ps.rsslim,
                    &ps.startcode,
                    &ps.endcode,
                    &ps.startstack,
                    &ps.kstkesp,
                    &ps.kstkeip,
                    &ps.signal,
                    &ps.blocked,
                    &ps.sigignore,
                    &ps.sigcatch,
                    &ps.wchan,
                    &ps.nswap,
                    &ps.cnswap,
                    &ps.exit_signal,
                    &ps.processor,
                    &ps.rt_priority,
                    &ps.policy,
                    &ps.delayacct_blkio_ticks,
                    &ps.guest_time,
                    &ps.cguest_time,
                    &ps.start_data,
                    &ps.end_data,
                    &ps.start_brk,
                    &ps.arg_start,
                    &ps.arg_end,
                    &ps.env_start,
                    &ps.env_end,
                    &ps.exit_code
    );
    return rv == 52;
}
