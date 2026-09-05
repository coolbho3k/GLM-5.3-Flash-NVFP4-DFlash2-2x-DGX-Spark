"""Probe-only backport of upstream EXL3's greedy expert assignment.

Reference: turboderp-org/exllamav3@499890c75d20d8e7c9d061f37189ae611a5c9f0b,
quant/exl3_moe_kernel.cuh. Keep our pinned GEMM implementation and launch ABI.
No serving patch: the isolated harness owns and zero-initializes its locks.
"""


def ticket_source(kernel):
    def once(old, new):
        nonlocal kernel
        if kernel.count(old) != 1:
            raise ValueError(f'Expected one ticket-scheduler anchor: {old!r}')
        kernel = kernel.replace(old, new, 1)

    once('    // Individual GEMM barriers per group', '''    // Probe lock allocation has MAX_TILES_C + 2048 ints. Reserve its upper
    // 1024 ints; the lower 1024 remain available for group barriers. The
    // launcher separately bounds the group count so these regions cannot overlap.
    int* sched = locks + BARRIER_LOCKS_OFFSET + 1024;
    int ticket = group_idx;

    // Individual GEMM barriers per group''')
    once('if (expert_idx_assign++ % concurrency != group_idx) continue;',
         'if (expert_idx_assign++ != ticket) continue;')
    once('''            group_barrier(group_idx, MOE_SMS_PER_EXPERT, barrier_counters_sense);
        };

        had_d_out();
    }
}''', '''        };

        had_d_out();
        // One leader claims work; the existing final barrier publishes the
        // group ticket and still orders all scatter writes before scratch reuse.
        if (block_idx == 0 && threadIdx.x == 0)
            sched[2 + group_idx] = concurrency + atomicAdd(sched, 1);
        group_barrier(group_idx, MOE_SMS_PER_EXPERT, barrier_counters_sense);
        ticket = sched[2 + group_idx];
    }
    // All groups retire, including those with no assigned experts. Acquire/
    // release retirement orders ticket increments before the last-group reset.
    // Stream ordering protects the next replay; workspace is never shared
    // between concurrent kernel launches in this harness.
    if (block_idx == 0 && threadIdx.x == 0)
    {
        cuda::atomic_ref<int, cuda::thread_scope_device> next(sched[0]);
        cuda::atomic_ref<int, cuda::thread_scope_device> retired(sched[1]);
        if (retired.fetch_add(1, cuda::memory_order_acq_rel) == concurrency - 1)
        {
            next.store(0, cuda::memory_order_relaxed);
            retired.store(0, cuda::memory_order_relaxed);
        }
    }
}''')
    return kernel
