"""Build the DEDICATED Drift MCDMA target and its client library from the owner's MIT-licensed MCDMA sources, in a directory
of Drift's own. Nothing is installed, nothing is started, the owner's binaries and services are not touched.

Differences from upstream `metal/mcdma_metal.c` (each applied by an asserted textual patch, so an upstream change fails loudly):
  1. its own UDP port (compile-time, --port), so it never shares an address with the owner's daemons;
  2. ONE region for every peer, backed by a file the caller names (argv[3]) and mapped MAP_SHARED, so the consumer on the same
     host maps the same bytes; upstream allocates a private heap region PER PEER IP, which a local consumer could never see;
  3. the region is not zeroed when a new peer appears (that would wipe a mailbox in flight);
  4. the per-write log line is verbose-only and no longer prints the first 48 bytes of the region: upstream logs every write
     under a global lock, which wedges the daemon when its stdout is an unread pipe and would copy mailbox bytes into logs;
  5. idle workers wake once a second, so SIGTERM stops the daemon; upstream blocks in recvfrom until a datagram arrives.
Usage: build.py --src <dir with mcdma_metal.c mcdma_wire.h libmcdma.c> --out <dir> --port N"""
import argparse, hashlib, platform, subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--src", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--port", type=int, required=True)
args = parser.parse_args()
if not 1024 < args.port < 65535 or args.port in (18777, 18555, 64087, 8888):
    raise SystemExit("choose an unprivileged port that the owner's services do not use")
args.out.mkdir(parents=True, exist_ok=True)


def patch(text, old, new, what):
    if text.count(old) != 1:
        raise SystemExit(f"upstream changed: cannot apply '{what}'")
    return text.replace(old, new)


wire = patch((args.src / "mcdma_wire.h").read_text(), "#define MCDMA_PORT      18777", "#ifndef MCDMA_PORT\n#define MCDMA_PORT      18777\n#endif", "port override guard")
daemon = (args.src / "mcdma_metal.c").read_text()
daemon = patch(daemon, "static size_t g_region_sz;", "static size_t g_region_sz;\nstatic void  *g_shared_region;   /* Drift: one file-backed region for every peer */", "shared region global")
daemon = patch(daemon, """    void *region = NULL;
    if (posix_memalign(&region, 16384, g_region_sz)) return NULL;
    memset(region, 0, g_region_sz);
    if (mlock(region, g_region_sz) != 0)
        fprintf(stderr, "warning: mlock failed (%s) — region not pinned\\n",
                strerror(errno));
""", "    void *region = g_shared_region;   /* Drift: shared, never zeroed here, pinned once in main */\n", "per-peer allocation")
for n, old in enumerate(("            munlock(region, g_region_sz); free(region);\n            return &peers[i];", "        munlock(region, g_region_sz); free(region);\n        return NULL;")):
    daemon = patch(daemon, old, old.replace("munlock(region, g_region_sz); free(region);", "/* Drift: the shared region is never freed per peer */"), f"free path {n}")
daemon = patch(daemon, "    g_region_sz        = mib << 20;\n", """    g_region_sz        = mib << 20;
    if (argc < 4 || argv[3][0] != '/') { fprintf(stderr, "usage: %s <bind-ip> <region-MiB> </absolute/region/file>\\n", argv[0]); return 2; }
    if (!strcmp(bindip, "0.0.0.0")) { fprintf(stderr, "refusing to bind every interface; name the link's address\\n"); return 2; }
    {
        int rfd = open(argv[3], O_RDWR | O_CREAT, 0600);
        if (rfd < 0 || ftruncate(rfd, (off_t)g_region_sz) != 0) { perror("region file"); return 1; }
        g_shared_region = mmap(NULL, g_region_sz, PROT_READ | PROT_WRITE, MAP_SHARED, rfd, 0);
        close(rfd);
        if (g_shared_region == MAP_FAILED) { perror("mmap region"); return 1; }
        if (mlock(g_shared_region, g_region_sz) != 0) fprintf(stderr, "warning: mlock failed (%s): region not pinned\\n", strerror(errno));
    }
""", "region file setup")
daemon = patch(daemon, '    printf("=== MCDMA METAL (Mac side) ===\\n");', '    printf("=== DRIFT MCDMA TARGET (dedicated; region file %s) ===\\n", argv[3]);', "banner")
daemon = patch(daemon, """                LOG("[PUT ] %s %llu segs, %llu bytes | text: %.48s\\n",
                    ip, p->nput, p->total_put, (char*)p->region);""", """                if (verbose) LOG("[PUT ] %s %llu segs, %llu bytes\\n", ip, p->nput, p->total_put);   /* Drift: off the hot path, and never print region bytes */""", "per-write log")
daemon = patch(daemon, """    if (bind(fd, (struct sockaddr*)&sa, sizeof sa)) { perror("bind"); return NULL; }
""", """    if (bind(fd, (struct sockaddr*)&sa, sizeof sa)) { perror("bind"); return NULL; }
    { struct timeval idle = { .tv_sec = 1 }; setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &idle, sizeof idle); }   /* Drift: an idle worker re-checks `running`, so SIGTERM stops the daemon */
""", "idle receive timeout")
daemon = patch(daemon, "#include <sys/mman.h>\n", "#include <sys/mman.h>\n#include <fcntl.h>\n", "fcntl include")
(args.out / "mcdma_wire.h").write_text(wire); (args.out / "drift_mcdma_target.c").write_text(daemon)
(args.out / "libmcdma.c").write_text((args.src / "libmcdma.c").read_text())
for name in ("mcdma.py", "LICENSE"):
    if (args.src / name).exists():
        (args.out / name).write_text((args.src / name).read_text())
lib = "libmcdma.dylib" if platform.system() == "Darwin" else "libmcdma.so"
flags = ["-O2", "-Wall", "-Wextra", f"-DMCDMA_PORT={args.port}", f"-I{args.out}"]
subprocess.run(["cc", *flags, "-pthread", "-o", str(args.out / "drift_mcdma_target"), str(args.out / "drift_mcdma_target.c")], check=True)
subprocess.run(["cc", *flags, *(["-dynamiclib"] if platform.system() == "Darwin" else ["-shared", "-fPIC"]), "-o", str(args.out / lib), str(args.out / "libmcdma.c")], check=True)
print({"port": args.port, "daemon": str(args.out / "drift_mcdma_target"), "library": str(args.out / lib),
       "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in (args.out / "drift_mcdma_target", args.out / lib)}})
