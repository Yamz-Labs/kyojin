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

          # 1. Load workspace (pyproject.toml + uv.lock at eval time)
          workspace = inputs.uv2nix.lib.workspace.loadWorkspace {
            workspaceRoot = ./.;
          };

          # 2. Overlay: uv.lock -> derivations (wheels preferred).
          # The cu124-cu132 extras are a uv `conflicts` group: uv2nix demands
          # exactly one selected branch per declaration (empty leaves all
          # torch forks in play -> "Non disjoint" solver error). Pin cu132
          # (newest, torch 2.14.1); the ROCm torch swap lands in 2b.
          uvLockedOverlay = workspace.mkPyprojectOverlay {
            sourcePreference = "wheel";
            dependencies = {
              exllamav3 = [ "cu132" ];
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
                autoPatchelfIgnoreMissingDeps = (old.autoPatchelfIgnoreMissingDeps or [ ]) ++ cudaIgnore;
              };
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
              # System headers for SDK clang (it runs outside the cc
              # wrapper). They must be searched AFTER clang's resource dir so
              # the cuda_wrappers stay engaged and their #include_next lands
              # in libstdc++/glibc, like /usr/include on Ubuntu.
              # Only -Xclang -internal-isystem lands after builtins
              # (-isystem/-idirafter/--sysroot never reach past the resource
              # dir, starving the wrappers). Verified via -### + live builds.
              hipSysIncludes = lib.concatStringsSep " " [
                "-Xclang -internal-isystem -Xclang ${pkgs.stdenv.cc.cc}/include/c++/${pkgs.stdenv.cc.cc.version}"
                "-Xclang -internal-isystem -Xclang ${pkgs.stdenv.cc.cc}/include/c++/${pkgs.stdenv.cc.cc.version}/${pkgs.stdenv.hostPlatform.config}"
                "-Xclang -internal-isystem -Xclang ${pkgs.stdenv.cc.libc.dev}/include"
              ];
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
            // {
              # ROCm torch is the default (kyojin is an AMD-first fork): swap
              # the wheel + version, link the SDK tree and ROCm math libs.
              # The CUDA sibling search/ignore from withCudaPatch is inert.
              torch = prev.torch.overrideAttrs (
                old:
                (withCudaPatch old)
                // {
                  inherit (amdTorch) version;
                  src = fetchAmdWheel {
                    inherit (amdTorch) url sha256;
                    name = "torch-${amdTorch.version}-cp312-cp312-linux_x86_64.whl";
                  };
                  buildInputs = (old.buildInputs or [ ]) ++ [
                    rocmSdk
                    rocmLibs
                  ];
                  nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [
                    pkgs.unzip
                  ];
                  # Merge the gfx1151 (+family) device kernel packs into
                  # torch's tree: torch looks up torch/.kpack/ + aotriton
                  # images relative to itself at runtime (strace-verified).
                  postFixup = (old.postFixup or "") + ''
                    for w in "${
                      fetchAmdWheel {
                        inherit (amdTorchDevice) url sha256;
                        name = "amd_torch_device_gfx1151-${amdTorchDevice.version}-cp312-cp312-linux_x86_64.whl";
                      }
                    }" "${
                      fetchAmdWheel {
                        inherit (amdTorchDeviceFamily) url sha256;
                        name = "amd_torch_device_gfx115x-${amdTorchDeviceFamily.version}-cp312-cp312-linux_x86_64.whl";
                      }
                    }"; do
                      unzip -q -o "$w" "torch/*" -d $TMPDIR/torch-device
                      cp -rn $TMPDIR/torch-device/torch/. $out/lib/python3.12/site-packages/torch/
                    done
                  '';
                }
              );
              # ROCm triton for the same reason (torch requires it; the CUDA
              # build's native lib hangs probing for NVIDIA hardware).
              triton = prev.triton.overrideAttrs (
                old:
                (withCudaPatch old)
                // {
                  inherit (amdTriton) version;
                  src = fetchAmdWheel {
                    inherit (amdTriton) url sha256;
                    name = "triton-${amdTriton.version}-cp312-cp312-linux_x86_64.whl";
                  };
                }
              );
              # Engine extension build: setup.py imports torch (cpp_extension)
              # and compiles HIP sources with the SDK toolchain. torch comes
              # from buildInputs (importable via PYTHONPATH in the pep517
              # build); providers must be importable too (torch's _rocm_init
              # preloads through them at import time).
              exllamav3 = prev.exllamav3.overrideAttrs (old: {
                buildInputs = (old.buildInputs or [ ]) ++ [
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
                nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [
                  rocmSdk
                ];
                preBuild = ''
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

          # 5. AMD nightly pins: multi-arch index, ROCm 10.1 bleeding edge
          # (uv-resolved `--pre` set, 2026-10-10; doc/install.md verifies
          # torch 2.14.0 on this ROCm, which is what we pin after 2.15.0a0
          # proved too raw for the ext build. To float torch again, bump
          # amdTorch only and re-verify `nix build .#exllamav3`).
          # Bump procedure: re-run the `uv pip compile` from the Chunk-1
          # notes against https://rocm.nightlies.amd.com/whl-multi-arch/ and
          # update url+sha256 below (nix-prefetch-url).
          # NOTE: keep `%2B` (not `+`) in wheel URLs: the index links are
          # percent-encoded and fetchurl must match byte-identically.
          amdIndex = "https://rocm.nightlies.amd.com/whl-multi-arch";
          amdTorch = {
            version = "2.14.0+rocm10.1.0a20260822";
            url = "${amdIndex}/torch-2.14.0%2Brocm10.1.0a20260822-cp312-cp312-linux_x86_64.whl";
            sha256 = "07v42ddqfm9dga0c09lxgmcfmx1zqbg01v9kzdnwl72l9dqq8svy";
          };
          # Device kernel packs torch looks up at runtime
          # (torch/.kpack/torch_gfx1151.kpack + aotriton images); merged
          # into torch's tree post-install, exactly as pip lays them out.
          amdTorchDevice = {
            version = "2.14.0+rocm10.1.0a20260822";
            url = "${amdIndex}/amd_torch_device_gfx1151-2.14.0%2Brocm10.1.0a20260822-cp312-cp312-linux_x86_64.whl";
            sha256 = "161a5rk2fwn3f6rv4z73z99ay50hz8r59bn3b8kb8i5v6s47a7k1";
          };
          amdTorchDeviceFamily = {
            version = "2.14.0+rocm10.1.0a20260822";
            url = "${amdIndex}/amd_torch_device_gfx115x-2.14.0%2Brocm10.1.0a20260822-cp312-cp312-linux_x86_64.whl";
            sha256 = "0chym3bv8y889cvkrpiv1cj3b8mf90dpw82y43qhnqz8kzn66z24";
          };
          # ROCm triton (torch Requires-Dist; the CUDA build hangs probing
          # for NVIDIA hardware on AMD boxes, and exllamav3 imports it).
          amdTriton = {
            version = "3.8.0+git675c5987.rocm10.1.0a20260822";
            url = "${amdIndex}/triton-3.8.0%2Bgit675c5987.rocm10.1.0a20260822-cp312-cp312-linux_x86_64.whl";
            sha256 = "06v9v8h0rqmw3jjnxlx434i1v6ml14pf2fqszhkrnq7vs381j6a2";
          };
          amdSdkDevel = {
            version = "10.1.0a20260822";
            url = "${amdIndex}/rocm_sdk_devel-10.1.0a20260822-py3-none-linux_x86_64.whl";
            sha256 = "1by8bs60zfn7gxk6yp6wdhzx14nl79rjzs0l6lhqlm97hmjiiqjn";
          };
          amdSdkLibraries = {
            version = "10.1.0a20260822";
            url = "${amdIndex}/rocm_sdk_libraries-10.1.0a20260822-py3-none-linux_x86_64.whl";
            sha256 = "1hnqsxqwfbm1ih4g4iv0g771vb0n4md1s9mgdixmbl5i06nbgw31";
          };
          amdSdkDevice = {
            version = "10.1.0a20260822";
            url = "${amdIndex}/rocm_sdk_device_gfx1151-10.1.0a20260822-py3-none-linux_x86_64.whl";
            sha256 = "0kk5llm72xdf3iy5wl5fx8nmyymg6l50gpjp1xjb0fqd104qi9in";
          };
          amdSdkCore = {
            version = "10.1.0a20260822";
            url = "${amdIndex}/rocm_sdk_core-10.1.0a20260822-py3-none-linux_x86_64.whl";
            sha256 = "0aap59x6l7yyzs7pfmlwszryl6f54j1w07hyb9shrzfgcls6sdyg";
          };
          # `rocm` meta sdist: provides the `rocm_sdk` registry module that
          # torch's _rocm_init imports (preload registry + version check).
          amdRocmSdist = {
            version = "10.1.0a20260822";
            url = "${amdIndex}/rocm-10.1.0a20260822.tar.gz";
            sha256 = "1k39cnidcn86p33hdiprzyngwj0rmplmnj8yagk1sg008mr0j5m7";
          };
          # `rocm-bootstrap`: platform/target detection used by the registry.
          amdBootstrap = {
            version = "0.1.0";
            url = "${amdIndex}/rocm_bootstrap-0.1.0-py3-none-any.whl";
            sha256 = "0lmn8jkd48vfcw0gz3vm5yvlykc7r8cacjf0dghg1cc6s3dd9qrk";
          };
          fetchAmdWheel =
            {
              url,
              sha256,
              name,
            }:
            pkgs.fetchurl { inherit url sha256 name; };
          develWheel = fetchAmdWheel {
            inherit (amdSdkDevel) url sha256;
            name = "rocm_sdk_devel-${amdSdkDevel.version}-py3-none-linux_x86_64.whl";
          };
          coreWheel = fetchAmdWheel {
            inherit (amdSdkCore) url sha256;
            name = "rocm_sdk_core-${amdSdkCore.version}-py3-none-linux_x86_64.whl";
          };
          librariesWheel = fetchAmdWheel {
            inherit (amdSdkLibraries) url sha256;
            name = "rocm_sdk_libraries-${amdSdkLibraries.version}-py3-none-linux_x86_64.whl";
          };
          deviceWheel = fetchAmdWheel {
            inherit (amdSdkDevice) url sha256;
            name = "rocm_sdk_device_gfx1151-${amdSdkDevice.version}-py3-none-linux_x86_64.whl";
          };
          bootstrapWheel = fetchAmdWheel {
            inherit (amdBootstrap) url sha256;
            name = "rocm_bootstrap-${amdBootstrap.version}-py3-none-any.whl";
          };
          rocmSdist = pkgs.fetchurl {
            inherit (amdRocmSdist) url sha256;
            name = "rocm-${amdRocmSdist.version}.tar.gz";
          };

          # 6. ROCm SDK tree (= `rocm-sdk init` output, sandbox-expanded).
          # Layout mirrors pip site-packages, which the wheels assume:
          # devel's lib/*.so are relative symlinks into a SIBLING
          # _rocm_sdk_core/ tree. Top-level compat links expose the paths
          # env.sh/build.sh expect (include/, bin/hipcc, lib/libhsa).
          # $out IS the SDK root (no `rocm-sdk path` CLI needed).
          rocmSdk = pkgs.stdenv.mkDerivation {
            pname = "rocm-sdk-gfx1151";
            inherit (amdSdkDevel) version;
            srcs = [
              develWheel
              coreWheel
            ];
            nativeBuildInputs = with pkgs; [
              unzip
              gnutar
              patchelf
            ];
            dontUnpack = true;
            installPhase = ''
              runHook preInstall
              work=$(mktemp -d)
              unzip -q -o "${develWheel}" -d $work/devel
              unzip -q -o "${coreWheel}" -d $work/core
              mkdir -p $out
              tar -xf $work/devel/rocm_sdk_devel/_devel.tar -C $out
              # core runtime tree as devel's sibling (devel's lib/bin links
              # expect ../../_rocm_sdk_core/); skip its python package bits.
              mkdir -p $out/_rocm_sdk_core
              for d in bin etc include lib libexec share; do
                [ -e $work/core/_rocm_sdk_core/$d ] && cp -r $work/core/_rocm_sdk_core/$d $out/_rocm_sdk_core/
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
            inherit (amdSdkLibraries) version;
            nativeBuildInputs = [ pkgs.unzip ];
            dontUnpack = true;
            # Upstream binaries: strip corrupts some (misaligned LOAD
            # segments after strip, e.g. librocsolver). Keep byte-identical.
            dontStrip = true;
            installPhase = ''
              runHook preInstall
              work=$(mktemp -d)
              mkdir -p $out/lib
              unzip -q -o "${librariesWheel}" "_rocm_sdk_libraries/lib/*" -d $work
              cp -r $work/_rocm_sdk_libraries/lib/. $out/lib/
              unzip -q -o "${deviceWheel}" "_rocm_sdk_libraries/lib/*" -d $work/device
              cp -rn $work/device/_rocm_sdk_libraries/lib/. $out/lib/
              unzip -q -o "${coreWheel}" "_rocm_sdk_core/lib/host-math/lib/*" "_rocm_sdk_core/lib/rocm_sysdeps/lib/*" -d $work
              cp -r $work/_rocm_sdk_core/lib/host-math/lib/. $out/lib/
              cp -r $work/_rocm_sdk_core/lib/rocm_sysdeps/lib/. $out/lib/
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
            inherit (amdRocmSdist) version;
            nativeBuildInputs = with pkgs; [
              unzip
              gnutar
            ];
            dontUnpack = true;
            # As in rocmLibs: upstream binaries, keep byte-identical.
            dontStrip = true;
            installPhase = ''
              runHook preInstall
              work=$(mktemp -d)
              sp=$out/lib/python3.12/site-packages
              mkdir -p $sp
              # provider layouts must stay native (rocm_sdk globs lib/ under
              # each importable package); core's bin/ stays out (its absolute
              # build-env symlinks don't survive outside site-packages).
              unzip -q -o "${coreWheel}" "_rocm_sdk_core/__init__.py" "_rocm_sdk_core/lib/*" -d $work/core
              cp -r $work/core/_rocm_sdk_core $sp/
              unzip -q -o "${librariesWheel}" "_rocm_sdk_libraries/__init__.py" "_rocm_sdk_libraries/lib/*" -d $work/libs
              cp -r $work/libs/_rocm_sdk_libraries $sp/
              unzip -q -o "${deviceWheel}" "_rocm_sdk_libraries/lib/*" -d $work/device
              cp -rn $work/device/_rocm_sdk_libraries/lib/. $sp/_rocm_sdk_libraries/lib/
              unzip -q -o "${bootstrapWheel}" "rocm_bootstrap/*" -d $work/bootstrap
              cp -r $work/bootstrap/rocm_bootstrap $sp/
              tar -xzf "${rocmSdist}" -C $work
              cp -r $work/rocm-${amdRocmSdist.version}/src/rocm_sdk $sp/
              runHook postInstall
            '';
          };

          # 9. kyojin serve package (house pattern 3: stdenv + makeWrapper).
          # Ships the three serve entry points with their sibling modules
          # (serve_metrics.py per flavor + shared startup_health.py) under
          # $out/share/kyojin/, mirroring the repo tools/ layout so the
          # scripts' sys.path bootstrap keeps working. Wrappers bake in the
          # engine venv python, provider/engine PYTHONPATH, EXL3_ROCM_SDK
          # and the Trap-1 LD_PRELOAD; CLI args pass straight through.
          kyojinServe = pkgs.stdenv.mkDerivation {
            pname = "kyojin";
            # Keep in sync with exllamav3/version.py (pyproject-nix reports
            # local-project version as 0.0.0 since it's setuptools-dynamic).
            version = "1.5.0";
            src = ./.;
            nativeBuildInputs = [ pkgs.makeWrapper ];
            installPhase = ''
              runHook preInstall
              mkdir -p $out/share/kyojin $out/bin
              for flavor in glm mimo qwen; do
                mkdir -p $out/share/kyojin/$flavor
                cp $src/tools/$flavor/serve.py $src/tools/$flavor/serve_metrics.py \
                  $out/share/kyojin/$flavor/
                # chat templates live next to the scripts (qwen/mimo) or in
                # lanes/assets (glm); the servers resolve them relative to
                # their own file, so ship them at the same relative paths.
                for tmpl in $src/tools/$flavor/chat_template.jinja; do
                  [ -e "$tmpl" ] && cp "$tmpl" $out/share/kyojin/$flavor/
                done
              done
              mkdir -p $out/share/kyojin/lanes/assets
              cp $src/tools/lanes/assets/glm53-template-medium.jinja \
                $out/share/kyojin/lanes/assets/
              cp $src/tools/startup_health.py $out/share/kyojin/
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

          # NOTE: pythonSet.torch IS the ROCm torch (swapped in myOverrides
          # above); packages.torch-rocm aliases it for a stable address.

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
            tag = "1.5.0";
            # Layer root is the cwd here: relative paths only. /tmp must
            # exist (Python tempfile, torch cuda shims); the model volume
            # supplies persistence, container /tmp stays ephemeral.
            extraCommands = ''
              mkdir -p tmp var/tmp etc
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
              # NOTE: no test deselection — every test_*_cpu.py file runs.
              # Two known-upstream issues live here (both fail identically on
              # main, which has no pytest CI): fake-hipcc scripts are POSIX sh
              # (fixed in-tree, was #!/bin/bash), and test_ablit_qwen_cpu.py
              # needs tools/ablit/* which was never committed (maintainer to
              # provide or update the test).
              files=$(ls tests/test_*_cpu.py)
              timeout 240 pytest $files -q
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
            hip-sysroot = hipSysroot;
            kyojin = kyojinServe;
            docker = kyojinImage;
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
              default = serveApp "glm";
            };
        };
    };
}
