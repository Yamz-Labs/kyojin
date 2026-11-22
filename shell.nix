# `nix-shell .` -> the flake's dev shell for this system.
#
# Same lock-derived flake-compat pin as default.nix. flake-compat returns the
# flake outputs, so the shell is addressed explicitly here rather than relying
# on a shim alias: exactly one definition of the dev shell lives in flake.nix.
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
) { src = ./.; }).shellNix.devShells.${builtins.currentSystem}.default
