#ifndef DRIFT_NATIVE_CONTRACT_H
#define DRIFT_NATIVE_CONTRACT_H
/* NEW bridge-owned interface proposal, NOT MCDMA's actual ABI.
 * Declarations only: M4 implements and tests a shim against pinned native APIs.
 * Region handles are process-local and never serialized as dereferenceable pointers.
 */
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef uint64_t tp_region;
typedef uint64_t tp_transfer;
typedef enum {
    TP_HOST_REGISTERED = 1,
    TP_METAL_SHARED = 2,
    TP_CUDA_MAPPED_HOST = 3
} tp_memory_kind;
typedef enum {
    TP_OK = 0, TP_PENDING = 1, TP_UNSUPPORTED = 2,
    TP_BAD_ARGUMENT = 3, TP_FAILED = 4, TP_IN_USE = 5
} tp_status;
typedef struct {
    uint32_t contract_version;
    tp_memory_kind kind;
    size_t bytes;
    size_t required_alignment;
} tp_allocation_request;
typedef struct {
    tp_region region;
    void *local_cpu_mapping;
    size_t bytes;
} tp_allocation;
typedef struct {
    tp_transfer transfer;
    uint64_t completed_bytes;
    uint32_t producer_visible;
    uint32_t receiver_visible;
    int32_t native_error;
} tp_completion;

/* Allocate memory that the actual GPU runtime and RDMA stack can both use. */
tp_status tp_allocate(const tp_allocation_request *, tp_allocation *);
/* Runtime-specific producer-completion fence. Must not report ready early. */
tp_status tp_producer_ready(tp_region, uint64_t epoch);
/* Peer region identifiers are validated capabilities from the run handshake. */
tp_status tp_submit(tp_region source, size_t source_offset,
                    uint64_t peer_region_capability, size_t peer_offset,
                    size_t bytes, tp_transfer *);
tp_status tp_poll(tp_transfer, tp_completion *);
/* Establish receiver GPU visibility before its consuming kernel may run. */
tp_status tp_receiver_acquire(tp_region, tp_transfer);
/* Refuse release while DMA work or GPU readers still own the region. */
tp_status tp_release(tp_region);
#ifdef __cplusplus
}
#endif
#endif
