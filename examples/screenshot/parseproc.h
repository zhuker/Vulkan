#pragma once
#include <cstdint>

namespace RBX::MemoryStats
{
    typedef uint64_t memsize_t;

    struct ProcMemInfo
    {
        memsize_t total;
        memsize_t free;
        memsize_t buffers;
        memsize_t cached;
        memsize_t memAvailable;
        memsize_t swapTotal;
        memsize_t swapFree;
    };

    struct ProcPidStatus
    {
        uint64_t vmSize{};
        uint64_t vmRss{};
        uint64_t vmSwap{};
        // Size of resident file mappings.  (since Linux 4.5).
        uint64_t rssFile{};
    };
#ifndef TASK_COMM_LEN
#define TASK_COMM_LEN 20
#endif
    struct ProcPidStat
    {
        int pid;
        char comm[TASK_COMM_LEN];
        char state;
        int ppid;
        int pgrp;
        int session;
        int tty_nr;
        int tpgid;
        unsigned flags;
        long unsigned minflt;
        long unsigned cminflt;
        // The number of major faults the process has made
        // which have required loading a memory page from disk
        long unsigned majflt;
        long unsigned cmajflt;
        long unsigned utime;
        long unsigned stime;
        long cutime;
        long cstime;
        long priority;
        long nice;
        long num_threads;
        long itrealvalue;
        long long unsigned starttime;
        // Virtual memory size in bytes.
        long unsigned vsize;

        // Resident Set Size: number of pages the process has
        // in real memory.  This is just the pages which count
        // toward text, data, or stack space.  This does not
        // include pages which have not been demand-loaded in,
        // or which are swapped out.  This value is
        // inaccurate; see /proc/pid/statm below.
        long rss;

        long unsigned rsslim;
        long unsigned startcode;
        long unsigned endcode;
        long unsigned startstack;
        long unsigned kstkesp;
        long unsigned kstkeip;
        long unsigned signal;
        long unsigned blocked;
        long unsigned sigignore;
        long unsigned sigcatch;
        long unsigned wchan;
        long unsigned nswap;
        long unsigned cnswap;
        int exit_signal;
        int processor;
        unsigned rt_priority;
        unsigned policy;
        long long unsigned delayacct_blkio_ticks;
        long unsigned guest_time;
        long cguest_time;
        long unsigned start_data;
        long unsigned end_data;
        long unsigned start_brk;
        long unsigned arg_start;
        long unsigned arg_end;
        long unsigned env_start;
        long unsigned env_end;
        int exit_code;
    };

    struct Vmstat
    {
        unsigned long pswpin;
        unsigned long pswpout;
    };

    bool parsePidStatus(const char* buf, RBX::MemoryStats::ProcPidStatus& status);

    //https://man7.org/linux/man-pages/man5/proc_pid_stat.5.html
    bool parsePidStat(const char* procstat, RBX::MemoryStats::ProcPidStat& ps);

    bool parseMemoryInfoBuf(char* buf, ProcMemInfo& meminfo);

    bool parseMemoryInfo(ProcMemInfo& meminfo);

    bool parseVmstat(const char *buf, Vmstat& vmstat);

}
