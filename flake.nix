{
  description = "kyojin - exllamav3 inference engine for AMD Strix Halo (gfx1151 / ROCm 7)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    flake-parts.url = "github:hercules-ci/flake-parts";

    git-hooks-nix = {
      url = "github:cachix/git-hooks.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs = {
        nixpkgs.follows = "nixpkgs";
        pyproject-nix.follows = "pyproject-nix";
      };
    };

    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs = {
        nixpkgs.follows = "nixpkgs";
        pyproject-nix.follows = "pyproject-nix";
        uv2nix.follows = "uv2nix";
      };
    };
  };

  outputs =
    inputs@{ flake-parts, ... }:
    # NOTE(3): re-add `self` to the pattern when apps.* reference self.packages.
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [ "x86_64-linux" ];

      imports = [ inputs.git-hooks-nix.flakeModule ];

      perSystem =
        { config, pkgs, ... }:
        let
          inherit (pkgs) lib;
          python = pkgs.python312;

          # System headers for SDK clang (it runs outside the cc
          # wrapper). They must be searched AFTER clang's resource dir so
          # the cuda_wrappers stay engaged and their #include_next lands
          # in libstdc++/glibc, like /usr/include on Ubuntu.
          # Only -Xclang -internal-isystem lands after builtins
          # (-isystem/-idirafter/--sysroot never reach past the resource
          # dir, starving the wrappers). Verified via -### + live builds.
          # Shared by the build-time exllamav3 compile AND the runtime JIT
          # wrappers below (gr_mix/gdn/ple/qsa/kda --genco on first serve):
          # the JIT needs the same sysincludes plus HIP_DEVICE_LIB_PATH
          # (Trap 4: bitcode lives at lib/llvm/amdgcn/bitcode, not the
          # $ROCM_PATH/amdgcn/bitcode clang probes).
          hipSysIncludes = lib.concatStringsSep " " [
            "-Xclang -internal-isystem -Xclang ${pkgs.stdenv.cc.cc}/include/c++/${pkgs.stdenv.cc.cc.version}"
            "-Xclang -internal-isystem -Xclang ${pkgs.stdenv.cc.cc}/include/c++/${pkgs.stdenv.cc.cc.version}/${pkgs.stdenv.hostPlatform.config}"
            "-Xclang -internal-isystem -Xclang ${pkgs.stdenv.cc.libc.dev}/include"
          ];

          # Narrowed source for the native extension compile: the only
          # repo files setup.py + torch's cpp_extension read (verified:
          # no #include escapes exllamav3_ext/, setup.py loads
          # build_config.py and util/arch_list.py by file path, the
          # version comes from an AST read of version.py, and README.md
          # feeds the wheel metadata). Python/docs/tests changes leave
          # this store path untouched, so the cached .so below survives
          # them (magic-nix-cache) instead of recompiling (~11 min).
          exllamaNativeSrc = lib.fileset.toSource {
            root = ./.;
            fileset = lib.fileset.unions [
              ./setup.py
              ./pyproject.toml
              ./README.md
              ./exllamav3/__init__.py
              ./exllamav3/version.py
              ./exllamav3/util/arch_list.py
              ./exllamav3/exllamav3_ext
            ];
          };

          # 1. Load workspace (pyproject.toml + uv.lock at eval time)
          workspace = inputs.uv2nix.lib.workspace.loadWorkspace {
            workspaceRoot = ./.;
          };

          # 2. Overlay: uv.lock -> derivations (wheels preferred).
          # The cu124-cu132+rocm extras are a uv `conflicts` group: uv2nix
          # demands exactly one selected branch per declaration (empty leaves
          # all torch forks in play -> "Non disjoint" solver error). Select
          # rocm (AMD nightly set, also pinned in pyproject.toml); CUDA
          # branches stay resolvable in uv for non-Nix flows.
          uvLockedOverlay = workspace.mkPyprojectOverlay {
            sourcePreference = "wheel";
            dependencies = {
              exllamav3 = [ "rocm" ];
            };
          };

          # 3. Python package set.
          # myOverrides: small house overlay for CUDA wheel linking.
          # The nvidia-* redistributables NEEDED-link each other across
          # separate derivations. Baking cross-wheel store paths at eval time
          # would recurse (each wheel's derivation feeding the others'), so:
          # - every wheel gets a static preFixup hook registering its
          #   buildInputs' lib dirs for auto-patchelf discovery at BUILD time
          #   (NIX_LDFLAGS is set by then; no derivation is interpolated);
          # - only torch (top of the stack, nothing references it back) takes
          #   the sibling wheels as buildInputs, giving it complete RUNPATHs;
          # - at RUNTIME the linker resolves the mid wheels' NEEDED
          #   transitively through torch's loaded libraries, with
          #   LD_LIBRARY_PATH (shell below) as backstop. Only libcuda.so.1
          #   (host driver) plus RDMA/HPC SONAMEs stay patch-time-ignored.
          # This overlay is extended (torch -> AMD nightly wheel).
          myOverrides =
            final: prev:
            let
              nvidiaSibs = builtins.filter (n: lib.hasPrefix "nvidia-" n) (builtins.attrNames prev);
              hpcLibs = with pkgs; [
                pkgs."rdma-core"
                libfabric
                openmpi
                pmix
                ucx
              ];
              cudaIgnore = [
                "libcuda.so.1"
                "libcudart.so.13"
                "libnvrtc.so.13"
                "libcupti.so.13"
                "libcufft.so.12"
                "libcurand.so.10"
                "libcusolver.so.12"
                "libcusolverMg.so.12"
                "libcusparse.so.12"
                "libcusparseLt.so.0"
                "libcublas.so.13"
                "libcublasLt.so.13"
                "libcufile.so.0"
                "libnvJitLink.so.13"
                "libnvtx.so.3"
                "libnccl.so.2"
                "libmlx5.so.1"
                "librdmacm.so.1"
                "libibverbs.so.1"
                "libmpi.so.40"
                "liboshmem.so.40"
                "libfabric.so.1"
                "libucs.so.0"
                "libucp.so.0"
                "libuct.so.0"
                "libucs_signal.so.0"
                "libpmix.so.2"
              ];
              ignoreCuda = old: {
                autoPatchelfIgnoreMissingDeps =
                  (old.autoPatchelfIgnoreMissingDeps or [ ]) ++ cudaIgnore ++ rocmIgnore;
              };
              # ROCm math/device SONAMEs, patch-time-ignored for torch (whose
              # own tree IS patched): the wheels NEEDED-link each other
              # across separate derivations whose lib dirs nest under
              # site-packages (invisible to auto-patchelf), and all of these
              # resolve at runtime from rocmLibs' flat lib/ tree plus the
              # RUNPATHs baked from torch's buildInputs.
              rocmIgnore = [
                "libMIOpen.so.1"
                "libamd_comgr.so.3"
                "librocblas.so.5"
                "libamdhip64.so.7"
                "librocm_sysdeps_sqlite3.so"
                "libamd_smi.so.27"
                "libcholmod.so.5"
                "libhiprtc.so.7"
                "libhsa-runtime64.so.1"
                "libomp.so"
                "librocm-core.so.1"
                "librocm_sysdeps_bz2.so"
                "librocm_sysdeps_z.so.1"
                "librocm_sysdeps_zstd.so.1"
                "librocprofiler-register.so.0"
                "libroctx64.so.4"
                # rocgdb-py3.10 (a debugger for another interpreter, never
                # executed here) links legacy libcrypt, absent everywhere.
                "libcrypt.so.1"
                # Other-interpreter Python bindings shipped in the core
                # wheel (ours is 3.12 and links fine); never loaded here.
                "libpython3.10.so.1.0"
                "libpython3.11.so.1.0"
                "libpython3.13.so.1.0"
                "libpython3.14.so.1.0"
              ];
              cudaSearchHook = old: {
                preFixup = (old.preFixup or "") + ''
                  # Register nested site-packages lib dirs (NIX_LDFLAGS only
                  # carries top-level lib/ dirs, which misses wheels nested
                  # under lib/pythonX.Y/site-packages). $buildInputs holds
                  # every input's store path; static script, no eval cycle.
                  for dep in $buildInputs; do
                    if [[ -d "$dep/lib/python3.12/site-packages" ]]; then
                      addAutoPatchelfSearchPath "$dep/lib/python3.12/site-packages"
                    fi
                  done
                '';
              };
              withCudaPatch = old: (ignoreCuda old) // (cudaSearchHook old);
              # hipSysIncludes lives at perSystem scope (shared with the
              # runtime JIT wrappers below); referenced as ${hipSysIncludes}.
              # Engine extension compile env, shared by the cached native
              # build (exllamav3-ext) and the pure wheel (exllamav3):
              # setup.py imports torch (cpp_extension) and compiles HIP
              # sources with the SDK toolchain. torch comes from
              # buildInputs (importable via PYTHONPATH in the pep517
              # build); providers must be importable too (torch's
              # _rocm_init preloads through them at import time).
              exllamaExtBuildInputs =
                old:
                (old.buildInputs or [ ])
                ++ [
                  final.torch
                  rocmSdk
                  # torch's own top-level imports (needed to import torch
                  # during setup.py's cpp_extension build).
                  final.filelock
                  final."typing-extensions"
                  final.sympy
                  final.networkx
                  final.jinja2
                  final.fsspec
                ];
              exllamaExtPreBuild = ''
                export HOME="$TMPDIR"
                echo "ENVDBG CPATH=$CPATH CPLUS=$CPLUS_INCLUDE_PATH CINC=$C_INCLUDE_PATH" >&2
                echo "ENVDBG NIX_CFLAGS=$NIX_CFLAGS_COMPILE" >&2
                # NOTE: export ROCM_HOME only (torch reads it for -I).
                # ROCM_PATH/HIP_PATH are deliberately UNSET: hipcc
                # self-locates its version + data files correctly without
                # them, but misdetects (HIP 3.5.0) when they point at a
                # tree whose share/hip/version it can't resolve.
                export ROCM_HOME="${rocmSdk}"
                unset ROCM_PATH HIP_PATH
                export HIP_DEVICE_LIB_PATH="${rocmSdk}/lib/llvm/amdgcn/bitcode"
                export PYTORCH_ROCM_ARCH="gfx1151"
                export EXL3_HIP_DEFINES="EXL3_HIP_STG_PAD"
                export PYTHONPATH="${rocmProviders}/lib/python3.12/site-packages:$PYTHONPATH"
                export LD_LIBRARY_PATH="${
                  lib.makeLibraryPath [
                    pkgs.gcc.cc.lib
                    rocmSdk
                    rocmLibs
                  ]
                }:$LD_LIBRARY_PATH"
                # SDK clang runs outside the cc wrapper: system headers for
                # host and device passes alike (plus NIX_CFLAGS_COMPILE's
                # python/SDK includes).
                export HIPCC_COMPILE_FLAGS_APPEND="${hipSysIncludes} $NIX_CFLAGS_COMPILE -v"
              '';
            in
            lib.genAttrs nvidiaSibs (
              n:
              prev.${n}.overrideAttrs (
                old:
                (withCudaPatch old)
                // {
                  buildInputs =
                    (old.buildInputs or [ ])
                    ++ lib.optionals (lib.hasInfix "nvshmem" n) hpcLibs
                    ++ lib.optionals (n == "nvidia-cufile" || n == "nvidia-cufile-cu12") [
                      pkgs."rdma-core"
                    ];
                }
              )
            )
            //
              lib.genAttrs
                [
                  "rocm-sdk-core"
                  "rocm-sdk-libraries"
                  "rocm-sdk-device-gfx1151"
                  "amd-torch-device-gfx1151"
                  "amd-torch-device-gfx115x"
                  # Triton ships prebuilt NVIDIA-backend blobs (never loaded
                  # on AMD) that auto-patchelf chokes on; same treatment.
                  "triton"
                ]
                (
                  n:
                  prev.${n}.overrideAttrs (_: {
                    # AMD's prebuilt binaries are fragile to ELF rewriting:
                    # patchelf-rewritten ones segfault on load (verified:
                    # identical wheel, patched tree crashes, pristine tree
                    # imports clean). Keep them byte-identical; at runtime
                    # they resolve via AMD's own $ORIGIN-relative RUNPATHs
                    # plus the venv co-installation, exactly as pip lays
                    # them out. (torch itself keeps its patch: proven green.)
                    dontAutoPatchelf = true;
                    dontStrip = true;
                  })
                )
            // {
              # The `rocm` meta sdist builds a trivial registry wheel but
              # needs setuptools/wheel, unfetchable in the sandbox and (as
              # a fetched package, unlike workspace members) not wired
              # into PYTHONPATH: export them explicitly.
              rocm = prev.rocm.overrideAttrs (old: {
                nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [
                  pkgs.python312Packages.setuptools
                  pkgs.python312Packages.wheel
                ];
                preBuild = (old.preBuild or "") + ''
                  export PYTHONPATH="${pkgs.python312Packages.setuptools}/lib/python3.12/site-packages:${pkgs.python312Packages.wheel}/lib/python3.12/site-packages:$PYTHONPATH"
                '';
              });
            }
            // {
              # ROCm torch resolves straight from the lock (the AMD nightly
              # set pinned in pyproject.toml): link the SDK tree and ROCm
              # math libs. The CUDA sibling search/ignore from withCudaPatch
              # is inert (no nvidia wheels in scope).
              torch = prev.torch.overrideAttrs (
                old:
                (withCudaPatch old)
                // {
                  buildInputs = (old.buildInputs or [ ]) ++ [
                    rocmSdk
                    rocmLibs
                  ];
                  # Merge the gfx1151 (+family) device kernel packs into
                  # torch's tree: torch looks up torch/.kpack/ + aotriton
                  # images relative to itself at runtime (strace-verified).
                  # Packs come from the lock, so versions always match torch.
                  postFixup = (old.postFixup or "") + ''
                    cp -rn ${
                      final."amd-torch-device-gfx1151"
                    }/lib/python3.12/site-packages/torch/. $out/lib/python3.12/site-packages/torch/
                    cp -rn ${
                      final."amd-torch-device-gfx115x"
                    }/lib/python3.12/site-packages/torch/. $out/lib/python3.12/site-packages/torch/
                  '';
                }
              );
              # Native extension, compiled ONLY from the narrowed native
              # sources (exllamaNativeSrc above). Its store path (and the
              # magic-nix-cache entry) is untouched by python/docs/tests
              # changes, so CI reuses the .so instead of recompiling.
              "exllamav3-ext" = prev.exllamav3.overrideAttrs (old: {
                src = exllamaNativeSrc;
                buildInputs = exllamaExtBuildInputs old;
                nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [
                  rocmSdk
                ];
                preBuild = exllamaExtPreBuild;
              });
              # Engine wheel: pure-python build (EXLLAMA_NOCOMPILE, seconds
              # even on a full-src rebuild) plus the cached .so injected
              # into site-packages, so ext.py finds the precompiled
              # extension at the same top-level import path as before.
              exllamav3 = prev.exllamav3.overrideAttrs (old: {
                buildInputs = exllamaExtBuildInputs old;
                nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [
                  rocmSdk
                ];
                preBuild = exllamaExtPreBuild + ''
                  # Pure wheel only: the native compile is skipped here;
                  # the .so comes from exllamav3-ext (postInstall below).
                  export EXLLAMA_NOCOMPILE=1
                '';
                postInstall = (old.postInstall or "") + ''
                  extSo=${final."exllamav3-ext"}/lib/python3.12/site-packages
                  copied=0
                  for f in "$extSo"/exllamav3_ext*.so; do
                    [ -e "$f" ] || continue
                    cp "$f" $out/lib/python3.12/site-packages/
                    copied=1
                  done
                  if [ "$copied" -eq 0 ]; then
                    echo "exllamav3: FATAL: no exllamav3_ext*.so in exllamav3-ext output ($extSo)" >&2
                    exit 1
                  fi
                '';
              });
            };
          pythonSet =
            (pkgs.callPackage inputs.pyproject-nix.build.packages { inherit python; }).overrideScope
              (
                lib.composeManyExtensions [
                  inputs.pyproject-build-systems.overlays.wheel
                  uvLockedOverlay
                  myOverrides
                ]
              );

          # Sysroot overlay: libstdc++ + glibc headers under usr/include,
          # searched AFTER clang's resource dir in every pass (host AND
          # device), exactly like /usr/include on Ubuntu. This keeps the
          # cuda_wrappers engaged (they #include_next into these) where
          # -isystem would bypass them or never reach the device pass.
          hipSysroot = pkgs.runCommand "hip-sysroot" { } ''
            mkdir -p $out/usr/include
            ln -s ${pkgs.stdenv.cc.cc}/include/c++ $out/usr/include/c++
            for f in ${pkgs.stdenv.cc.libc.dev}/include/*; do
              ln -s "$f" $out/usr/include/
            done
          '';

          # 4. Runtime env from the lock with ROCm torch from myOverrides.
          # CUDA extras stay resolvable in uv for non-Nix flows.
          kyojinEnv = pythonSet.mkVirtualEnv "kyojin-env" workspace.deps.default;

          # Library dirs backing the patch-time ignore list above: the CUDA
          # wheels' lib dirs plus the RDMA/HPC system libs.
          cudaLibPath = lib.makeLibraryPath (
            map (n: pythonSet.${n}) (
              builtins.filter (n: lib.hasPrefix "nvidia-" n) (builtins.attrNames pythonSet)
            )
            ++ (with pkgs; [
              pkgs."rdma-core"
              libfabric
              openmpi
              pmix
              ucx
            ])
          );

          # 5. ROCm versions are pinned once, in pyproject.toml (the `rocm`
          # extra + AMD index): uv.lock carries the wheel URLs/hashes, and
          # the derivations below consume the locked wheels via pythonSet.
          # Bump procedure: update the pins in pyproject.toml, re-run
          # `uv lock`, re-verify `nix build .#exllamav3`.

          # 6. ROCm SDK tree (= `rocm-sdk init` output, sandbox-expanded).
          # Layout mirrors pip site-packages, which the wheels assume:
          # devel's lib/*.so are relative symlinks into a SIBLING
          # _rocm_sdk_core/ tree. Top-level compat links expose the paths
          # env.sh/build.sh expect (include/, bin/hipcc, lib/libhsa).
          # $out IS the SDK root (no `rocm-sdk path` CLI needed).
          rocmSdk = pkgs.stdenv.mkDerivation {
            pname = "rocm-sdk-gfx1151";
            version = pythonSet."rocm-sdk-devel".version;
            nativeBuildInputs = with pkgs; [
              gnutar
              patchelf
            ];
            dontUnpack = true;
            installPhase = ''
              runHook preInstall
              devel=${pythonSet."rocm-sdk-devel"}/lib/python3.12/site-packages/rocm_sdk_devel
              core=${pythonSet."rocm-sdk-core"}/lib/python3.12/site-packages/_rocm_sdk_core
              mkdir -p $out
              tar -xf $devel/_devel.tar -C $out
              # core runtime tree as devel's sibling (devel's lib/bin links
              # expect ../../_rocm_sdk_core/); skip its python package bits.
              mkdir -p $out/_rocm_sdk_core
              for d in bin etc include lib libexec share; do
                [ -e $core/$d ] && cp -r $core/$d $out/_rocm_sdk_core/
              done
              # Repair AMD's absolute build-env symlinks to our sibling tree.
              find $out -xtype l | while read -r l; do
                t=$(readlink "$l")
                case "$t" in
                  /nix/store/_rocm_sdk_core/*)
                    rel=''${t#/nix/store/_rocm_sdk_core/}
                    if [ -e "$out/_rocm_sdk_core/$rel" ]; then
                      ln -sfn "$(realpath --relative-to="$(dirname "$l")" "$out/_rocm_sdk_core/$rel")" "$l"
                    fi
                    ;;
                esac
              done
              ln -s _rocm_sdk_devel/include $out/include
              ln -s _rocm_sdk_devel/bin $out/bin
              ln -s _rocm_sdk_devel/lib $out/lib
              ln -s _rocm_sdk_devel/llvm $out/llvm
              # hipcc resolves its version + data files relative to
              # ROCM_PATH/HIP_PATH when set (env.sh sets both): expose them.
              ln -s _rocm_sdk_devel/share $out/share
              ln -s _rocm_sdk_devel/etc $out/etc
              ln -s _rocm_sdk_devel/libexec $out/libexec
              ln -s _rocm_sdk_devel/amdgcn $out/amdgcn
              runHook postInstall
            '';
            # The upstream tree ships dangling test-fixture symlinks
            # (roctracer golden traces, therock manifests); never traversed
            # at build/serve time, so keep the tree byte-identical.
            dontCheckForBrokenSymlinks = true;
            # Prebuilt toolchain: point the loader at nixpkgs glibc and add
            # system lib dirs to executables' RUNPATHs (AMD $ORIGIN entries
            # are preserved). Libraries need no INTERP; their system deps
            # resolve through the executables' RUNPATHs at load time.
            postFixup = ''
              sysRpath="${pkgs.gcc.cc.lib}/lib:${pkgs.ncurses}/lib:${pkgs.libxml2}/lib:${pkgs.zlib}/lib"
              for d in $out/_rocm_sdk_devel/bin $out/_rocm_sdk_devel/llvm/bin $out/_rocm_sdk_devel/libexec $out/_rocm_sdk_core/bin; do
                [ -d "$d" ] || continue
                for f in "$d"/*; do
                  [ -f "$f" ] && [ -x "$f" ] || continue
                  head -c4 "$f" | grep -q $'\x7fELF' || continue
                  patchelf --set-interpreter "$(cat $NIX_CC/nix-support/dynamic-linker)" "$f"
                  oldRpath=$(patchelf --print-rpath "$f" 2>/dev/null || true)
                  patchelf --set-rpath "''${oldRpath:+$oldRpath:}$sysRpath" "$f"
                done
              done
            '';
            meta = {
              description = "AMD ROCm SDK (gfx1151 nightly) with devel headers";
            };
          };

          # 7. ROCm math/device libraries (torch's `rocm[libraries]` extra,
          # gfx1151 device kernels, plus two sidecar dirs torch links:
          # host-math OpenBLAS and the vendored sysdeps, both from the core
          # wheel). Flat lib/ tree consumed via buildInputs' lib dirs. Kept
          # surgical (not all of core's lib/) so devel's libhsa etc. always
          # win on SONAMEs.
          rocmLibs = pkgs.stdenv.mkDerivation {
            pname = "rocm-sdk-libraries-gfx1151";
            version = pythonSet."rocm-sdk-libraries".version;
            dontUnpack = true;
            # Upstream binaries: strip corrupts some (misaligned LOAD
            # segments after strip, e.g. librocsolver). Keep byte-identical.
            dontStrip = true;
            installPhase = ''
              runHook preInstall
              mkdir -p $out/lib
              libs=${pythonSet."rocm-sdk-libraries"}/lib/python3.12/site-packages/_rocm_sdk_libraries/lib
              cp -r $libs/. $out/lib/
              device=${pythonSet."rocm-sdk-device-gfx1151"}/lib/python3.12/site-packages/_rocm_sdk_libraries/lib
              cp -rn $device/. $out/lib/
              core=${pythonSet."rocm-sdk-core"}/lib/python3.12/site-packages/_rocm_sdk_core
              # comgr + hip live only in core's lib/ (not the sidecars
              # below); device libraries NEEDED-link them, so they join
              # the flat tree like everything else.
              cp -r $core/lib/libamd_comgr* $core/lib/libamdhip64* $out/lib/
              cp -r $core/lib/host-math/lib/. $out/lib/
              cp -r $core/lib/rocm_sysdeps/lib/. $out/lib/
              runHook postInstall
            '';
          };

          # 8. ROCm provider python packages: the native layout torch's
          # _rocm_init requires (importable _rocm_sdk_core/ +
          # _rocm_sdk_libraries/ (libraries+device wheels merged, as pip
          # does) + rocm_sdk/ registry + rocm_bootstrap/ detector).
          # Wheels/sdist preserved verbatim; composed via PYTHONPATH.
          rocmProviders = pkgs.stdenv.mkDerivation {
            pname = "rocm-python-providers";
            version = pythonSet.rocm.version;
            nativeBuildInputs = [ pkgs.gnutar ];
            dontUnpack = true;
            # As in rocmLibs: upstream binaries, keep byte-identical.
            dontStrip = true;
            installPhase = ''
              runHook preInstall
              core=${pythonSet."rocm-sdk-core"}/lib/python3.12/site-packages/_rocm_sdk_core
              libs=${pythonSet."rocm-sdk-libraries"}/lib/python3.12/site-packages/_rocm_sdk_libraries
              device=${pythonSet."rocm-sdk-device-gfx1151"}/lib/python3.12/site-packages/_rocm_sdk_libraries
              bootstrap=${pythonSet."rocm-bootstrap"}/lib/python3.12/site-packages/rocm_bootstrap
              sp=$out/lib/python3.12/site-packages
              mkdir -p $sp
              # provider layouts must stay native (rocm_sdk globs lib/ under
              # each importable package); core's bin/ stays out (its absolute
              # build-env symlinks don't survive outside site-packages).
              mkdir -p $sp/_rocm_sdk_core $sp/_rocm_sdk_libraries
              cp $core/__init__.py $sp/_rocm_sdk_core/
              cp -r $core/lib $sp/_rocm_sdk_core/
              cp $libs/__init__.py $sp/_rocm_sdk_libraries/
              cp -r $libs/lib $sp/_rocm_sdk_libraries/
              # cp preserves the store's read-only modes; the device merge
              # below writes into this tree, so make it writable first.
              chmod -R u+w $sp/_rocm_sdk_libraries
              cp -rn $device/lib/. $sp/_rocm_sdk_libraries/lib/
              cp -r $bootstrap $sp/
              # Registry module straight from the sdist tarball (building
              # the sdist wheel would need setuptools with no network).
              work=$(mktemp -d)
              tar -xzf ${pythonSet.rocm.src} -C $work
              cp -r $work/rocm-*/src/rocm_sdk $sp/
              runHook postInstall
            '';
          };

          # 9. kyojin serve package (house pattern 3: stdenv + makeWrapper).
          # Docker parity: $out/share/kyojin IS tools/ (full tree, zero
          # curation), mirroring docker/Dockerfile's COPY . . so the scripts'
          # sys.path bootstrap keeps working. Never an allowlist: past
          # outages (spec_policy, image_cache, serve_compat) all came from
          # a new sibling import missing its cp line.
          # Wrappers bake in the
          # engine venv python, provider/engine PYTHONPATH, EXL3_ROCM_SDK
          # and the Trap-1 LD_PRELOAD; CLI args pass straight through.
          # Runtime JIT (gr_mix/gdn/ple/qsa/kda --genco on first serve)
          # needs the same compiler env as the build: HIP_DEVICE_LIB_PATH
          # (Trap 4 bitcode) + HIPCC_COMPILE_FLAGS_APPEND (sysincludes).
          # Compiler-only: LD_LIBRARY_PATH stays SDK-free per Trap 5.
          kyojinServe = pkgs.stdenv.mkDerivation {
            pname = "kyojin";
            # Keep in sync with exllamav3/version.py (pyproject-nix reports
            # local-project version as 0.0.0 since it's setuptools-dynamic).
            version = "1.5.8";
            src = ./.;
            nativeBuildInputs = [ pkgs.makeWrapper ];
            installPhase = ''
              runHook preInstall
              mkdir -p $out/share/kyojin $out/bin
              # Docker parity (never allowlist): $out/share/kyojin IS tools/.
              # serve.py resolves siblings via its own dir (script-dir) +
              # tools/ (HERE.parent), and glm additionally appends ../qwen
              # for spec_policy — all preserved by construction. -L
              # dereferences the ../strix_halo symlinks (image_cache,
              # serve_metrics). No curation: every future sibling ships.
              cp -rL $src/tools/* $out/share/kyojin/
              rm -rf $out/share/kyojin/__pycache__ $out/share/kyojin/*/__pycache__
              # startup_health loads exllamav3.util.hip_compiler by path
              # (without importing exllamav3, to avoid latching engine env);
              # mirror the repo-root layout it expects.
              mkdir -p $out/share/exllamav3/util
              cp $src/exllamav3/util/hip_compiler.py $out/share/exllamav3/util/
              for flavor in glm mimo qwen; do
                makeWrapper ${kyojinEnv}/bin/python $out/bin/kyojin-serve-$flavor \
                  --add-flags "$out/share/kyojin/$flavor/serve.py" \
                  --prefix PYTHONPATH : "${pythonSet.exllamav3}/lib/python3.12/site-packages:${rocmProviders}/lib/python3.12/site-packages:$out/share/kyojin" \
                  --set EXL3_ROCM_SDK "${rocmSdk}" \
                  --set EXL3_HIPCC "${rocmSdk}/bin/hipcc" \
                  --set HIP_DEVICE_LIB_PATH "${rocmSdk}/lib/llvm/amdgcn/bitcode" \
                  --set HIPCC_COMPILE_FLAGS_APPEND "${hipSysIncludes}" \
                  --prefix LD_LIBRARY_PATH : "${pkgs.gcc.cc.lib}/lib:${cudaLibPath}" \
                  --set LD_PRELOAD "${rocmSdk}/lib/libhsa-runtime64.so.1"
              done
              runHook postInstall
            '';
            meta = {
              description = "kyojin exllamav3 inference server (AMD gfx1151)";
              mainProgram = "kyojin-serve-glm";
            };
          };

          # NOTE: pythonSet.torch IS the ROCm torch (lock resolution);
          # packages.torch-rocm aliases it for a stable address.

          # 9b. kyojin benchmark runner (house pattern 3: stdenv + makeWrapper).
          # bench.py is stdlib-only, so plain python3 suffices (no engine
          # venv), and it runs unmodified: default corpus is the repo's own
          # Markdown (run it from a checkout, e.g. nix develop), or pass
          # --corpus explicitly anywhere else.
          kyojinBench = pkgs.stdenv.mkDerivation {
            pname = "kyojin-bench";
            version = "1.5.8";
            src = ./.;
            nativeBuildInputs = [ pkgs.makeWrapper ];
            installPhase = ''
              runHook preInstall
              mkdir -p $out/share/kyojin $out/bin
              cp $src/tools/bench.py $out/share/kyojin/
              makeWrapper ${pkgs.python312}/bin/python3 $out/bin/kyojin-bench \
                --add-flags "$out/share/kyojin/bench.py"
              runHook postInstall
            '';
            meta = {
              description = "kyojin benchmark runner (issue #1 reports)";
              mainProgram = "kyojin-bench";
            };
          };

          # Pinned benchmark corpus (optional companion to kyojin-bench):
          # every repo Markdown file concatenated at build time, so reports
          # stay comparable between machines. bench.py's own default is
          # whatever markdown surrounds it at runtime; pass this file as
          # --corpus to freeze the input instead.
          benchCorpus = pkgs.runCommand "kyojin-bench-corpus" { src = ./.; } ''
            mkdir -p $out
            find $src -name '*.md' -not -path '*/.git/*' -not -path '*/.venv/*' -not -path '*/node_modules/*' -print0 \
              | sort -z | xargs -0 cat > $out/corpus.txt
          '';

          # 10. Container image + entrypoint. Ships the full SDK + toolchain
          # because serve builds GPU kernels on first use (install.md) with
          # hipcc; a runtime-lib subset would break that. A later diet to a
          # runtime subset is possible once the first-launch compile set is
          # characterized.
          # Model packs are NEVER baked in: the entrypoint downloads the
          # configured repo into /models (a volume) on first start.
          kyojinEntryPoint = pkgs.writeShellScript "kyojin-entrypoint" ''
            set -euo pipefail
            flavor="''${KYOJIN_FLAVOR:-}"
            case "$flavor" in
              qwen | glm | mimo) ;;
              *)
                echo "kyojin-entrypoint: set KYOJIN_FLAVOR to one of: qwen glm mimo" >&2
                exit 2
                ;;
            esac
            repo="''${KYOJIN_MODEL_REPO:-}"
            if [ -z "$repo" ]; then
              echo "kyojin-entrypoint: set KYOJIN_MODEL_REPO (e.g. yamz-labs/GLM-5.3-Flash-EXL3-Yamz)" >&2
              exit 2
            fi
            model_dir="''${KYOJIN_MODEL_DIR:-/models/$flavor}"
            # Tuning cache (e.g. GLM's dense GEMM tune, ~8 min first launch)
            # must survive container replacement: keep HOME on the volume.
            export HOME="''${KYOJIN_HOME:-/models/home}"
            mkdir -p "$HOME" "$model_dir"
            # One slot per flavor; the marker records which repo filled it so
            # switching repos under the same flavor fails loudly instead of
            # serving a stale pack.
            marker="$model_dir/.kyojin-repo"
            if [ -f "$model_dir/config.json" ]; then
              if [ -f "$marker" ] && [ "$(cat "$marker")" != "$repo" ]; then
                echo "kyojin-entrypoint: $model_dir holds $(cat "$marker"), not $repo;" >&2
                echo "  set KYOJIN_MODEL_DIR or empty the directory" >&2
                exit 2
              fi
              echo "kyojin-entrypoint: model present at $model_dir, skipping download" >&2
            else
              echo "kyojin-entrypoint: downloading $repo into $model_dir (first start is slow)" >&2
              hf download "$repo" --local-dir "$model_dir"
              printf '%s' "$repo" > "$marker"
            fi
            echo "kyojin-entrypoint: serving $flavor from $model_dir" >&2
            exec "''${KYOJIN_BIN:-/bin/kyojin-serve-$flavor}" --model "$model_dir" "$@"
          '';
          # huggingface-hub CLI (hf download for the entrypoint). nixpkgs'
          # anyio runs a networked TLS test suite in installCheck that fails
          # in the sandbox; disable it (self-contained CLI, version skew vs
          # the app env is irrelevant).
          huggingfaceHub =
            let
              pyForHub = pkgs.python312.override {
                packageOverrides = _self: super: {
                  anyio = super.anyio.overrideAttrs (_: {
                    doInstallCheck = false;
                  });
                };
              };
            in
            pyForHub.pkgs.huggingface-hub;
          kyojinImage = pkgs.dockerTools.buildLayeredImage {
            name = "kyojin";
            tag = "1.5.8";
            # Layer root is the cwd here: relative paths only. /tmp must
            # exist (Python tempfile, torch cuda shims); the model volume
            # supplies persistence, container /tmp stays ephemeral.
            extraCommands = ''
              mkdir -p tmp var/tmp etc
              # Real /tmp semantics: the container user is not root
              # (userNS keep-id), so layer-created tmp dirs must be
              # world-writable-sticky, or nothing can create temp files.
              chmod 1777 tmp var/tmp
            '';
            contents = [
              kyojinServe
              kyojinEnv
              rocmProviders
              rocmLibs
              rocmSdk
              pkgs.gcc
              pkgs.ninja
              pkgs.bashInteractive
              pkgs.coreutils
              pkgs.cacert
              pkgs.curl
              pkgs.dockerTools.caCertificates
              huggingfaceHub
            ];
            config = {
              Entrypoint = [ "${kyojinEntryPoint}" ];
              Cmd = [
                "--port"
                "8000"
              ];
              ExposedPorts = {
                "8000/tcp" = { };
              };
              Volumes = {
                "/models" = { };
              };
              Env = [
                "PATH=/bin:/usr/bin"
                "SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"
                "NIX_SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"
                "REQUESTS_CA_BUNDLE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"
              ];
            };
          };
        in
        {
          pre-commit.settings.hooks = {
            nixfmt.enable = true;
            deadnix.enable = true;
            statix.enable = true;
            # Python hooks (ruff etc.) after the first fmt pass:
            # this tree has no lint history, so start nix-only to keep
            # `nix flake check` green, then expand.
          };

          devShells.default = pkgs.mkShell {
            packages = [
              kyojinEnv
              pkgs.uv
              pkgs.ruff
              pkgs.ninja
              pkgs.gcc
              rocmSdk
            ]
            ++ config.pre-commit.settings.enabledPackages;
            shellHook = ''
              ${config.pre-commit.installationScript}
              export EXL3_ROOT="$PWD"
              export PYTHONPATH="$PWD:${rocmProviders}/lib/python3.12/site-packages:$PYTHONPATH"
              # ROCm runtime needs gcc libs at dlopen time (ctypes preloads).
              export LD_LIBRARY_PATH="${cudaLibPath}:${pkgs.gcc.cc.lib}/lib:''${LD_LIBRARY_PATH:-}"
              export EXL3_ROCM_SDK="${rocmSdk}"
              # Runtime JIT needs the same compiler env as the serve wrappers
              # (Trap 4 bitcode + sysincludes); compiler-only, no loader effect.
              export HIP_DEVICE_LIB_PATH="${rocmSdk}/lib/llvm/amdgcn/bitcode"
              export HIPCC_COMPILE_FLAGS_APPEND="${hipSysIncludes}"
            '';
          };

          # CI check `checks.<sys>.pre-commit` is provided by the flakeModule
          # itself (gated by pre-commit.check.enable, default true).

          # CPU test subset (GPU tests can't run here; GPU-touching imports
          # hang on non-gfx1151 hosts, so GPUs are hidden and the run is
          # time-bounded, failing loudly instead of hanging CI).
          checks.kyojin-tests = pkgs.stdenv.mkDerivation {
            pname = "kyojin-tests";
            version = "0";
            src = ./.;
            buildInputs = [ kyojinEnv ];
            nativeBuildInputs = [
              pkgs.coreutils
              # pytest is not in uv.lock (runner only, version-independent).
              pkgs.python312Packages.pytest
            ];
            doCheck = true;
            dontBuild = true;
            checkPhase = ''
              runHook preCheck
              export PYTHONPATH="${pythonSet.exllamav3}/lib/python3.12/site-packages:${rocmProviders}/lib/python3.12/site-packages:$PYTHONPATH"
              export LD_LIBRARY_PATH="${cudaLibPath}:${pkgs.gcc.cc.lib}/lib:''${LD_LIBRARY_PATH:-}"
              export EXL3_ROCM_SDK="${rocmSdk}"
              export ROCR_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" CUDA_VISIBLE_DEVICES=""
              # NOTE: every test_*_cpu.py file runs except test_ablit_qwen_cpu.py,
              # which upstream is removing (its tools/ablit/* helpers were
              # never committed anywhere; it fails identically on main,
              # which has no pytest CI). Left out of the check until that
              # lands; harmless afterwards (grep matches nothing).
              files=$(ls tests/test_*_cpu.py | grep -v test_ablit_qwen_cpu)
              timeout 240 pytest $files -q
              runHook postCheck
            '';
            installPhase = ''
              runHook preInstall
              touch $out
              runHook postInstall
            '';
          };

          # JIT smoke: cross-compile a dummy HIP kernel plus the real gr_mix.hip
          # with the same compiler env the serve wrappers set. Pure
          # cross-compilation (--genco --offload-arch): no GPU/KFD needed, so
          # this runs on any CI runner. Catches a missing HIP_DEVICE_LIB_PATH
          # (Trap 4: "cannot find ROCm device library") or missing sysincludes
          # ("Could not find standard C++ header") without ever serving.
          # The wrapper grep pins the two vars to kyojinServe so they cannot
          # silently drift apart from what is tested here.
          checks.kyojin-hipcc-smoke = pkgs.stdenv.mkDerivation {
            pname = "kyojin-hipcc-smoke";
            version = "0";
            src = ./.;
            dontBuild = true;
            doCheck = true;
            checkPhase = ''
              runHook preCheck
              export HIP_DEVICE_LIB_PATH="${rocmSdk}/lib/llvm/amdgcn/bitcode"
              export HIPCC_COMPILE_FLAGS_APPEND="${hipSysIncludes}"
              printf '__global__ void dummy() {}\n' > dummy.hip
              "${rocmSdk}/bin/hipcc" --genco --offload-arch=gfx1151 -O3 -o dummy.hsaco dummy.hip
              test -s dummy.hsaco
              "${rocmSdk}/bin/hipcc" --genco --offload-arch=gfx1151 -O3 -ffp-contract=off \
                -o gr_mix.hsaco exllamav3/modules/gr_fn/gr_mix.hip
              test -s gr_mix.hsaco
              for flavor in glm mimo qwen; do
                grep -q HIP_DEVICE_LIB_PATH ${kyojinServe}/bin/kyojin-serve-$flavor
                grep -q HIPCC_COMPILE_FLAGS_APPEND ${kyojinServe}/bin/kyojin-serve-$flavor
              done
              runHook postCheck
            '';
            installPhase = ''
              runHook preInstall
              touch $out
              runHook postInstall
            '';
          };

          formatter = pkgs.writeShellScriptBin "fmt" ''
            ${lib.getExe config.pre-commit.settings.package} run --all-files --config ${config.pre-commit.settings.configFile}
          '';

          packages = {
            default = kyojinEnv;
            torch-rocm = pythonSet.torch;
            rocm-sdk = rocmSdk;
            rocm-libs = rocmLibs;
            rocm-providers = rocmProviders;
            inherit (pythonSet) exllamav3;
            # Native extension alone (narrowed sources, cached across
            # python-only changes); the .so is injected into .#exllamav3.
            exllama-ext = pythonSet."exllamav3-ext";
            hip-sysroot = hipSysroot;
            kyojin = kyojinServe;
            docker = kyojinImage;
            bench = kyojinBench;
            bench-corpus = benchCorpus;
          };

          apps =
            let
              serveApp = flavor: {
                type = "app";
                program = "${kyojinServe}/bin/kyojin-serve-${flavor}";
              };
            in
            {
              kyojin-serve-qwen = serveApp "qwen";
              kyojin-serve-glm = serveApp "glm";
              kyojin-serve-mimo = serveApp "mimo";
              kyojin-bench = {
                type = "app";
                program = "${kyojinBench}/bin/kyojin-bench";
              };
              default = serveApp "glm";
            };
        };
    };
}
