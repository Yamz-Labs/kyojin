# Non-flake entry point, derived from flake.lock so it never needs a manual
# rev or hash. `nix-build -A kyojin .` and `nix-build .` both work here.
#
# flake-compat is pinned in flake.lock (flake = false), so this file and
# shell.nix stay in step with the flake without duplicating a revision.
#
# The flake nests its packages under `packages.<system>`, which is a detail a
# `nix-build` user should not have to know: the packages of this system are
# flattened to the top level here, so `nix-build -A kyojin .` selects the same
# derivation as `.#packages.<system>.kyojin`. Only the flake-compat view is
# flattened; the flake's own outputs are untouched.
let
  flake =
    (import (
      let
        lock = builtins.fromJSON (builtins.readFile ./flake.lock);
        inherit (lock.nodes.${lock.nodes.root.inputs.flake-compat}.locked)
          narHash
          rev
          ;
      in
      fetchTarball {
        url = "https://github.com/NixOS/flake-compat/archive/${rev}.tar.gz";
        sha256 = narHash;
      }
    ) { src = ./.; }).defaultNix;
  systemPackages = flake.packages.${builtins.currentSystem} or { };
in
flake // systemPackages
