{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    systems.url = "github:nix-systems/default";
    treefmt-nix = {
      url = "github:numtide/treefmt-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    flake-compat = {
      url = "github:edolstra/flake-compat";
      flake = false;
    };
  };

  outputs = { self, nixpkgs, systems, treefmt-nix, ... }:
    let
      eachSystem = f: nixpkgs.lib.genAttrs (import systems) (system: f
        (import nixpkgs {
          inherit system;
        })
      );
      treefmtEval = pkgs: (treefmt-nix.lib.evalModule pkgs (_: {
        projectRootFile = "flake.nix";
        programs = {
          black.enable = true;
          deadnix.enable = true;
          mdformat.enable = true;
          nixpkgs-fmt.enable = true;
          statix.enable = true;
        };
        settings.global.excludes = [ ".envrc" ];
      })).config.build.wrapper;
    in
    {
      packages = eachSystem (pkgs:
        let
          python3Packages = pkgs.python3.pkgs;
        in
        rec {
          auto-stack = python3Packages.buildPythonApplication {
            name = "auto-stack";
            src = ./.;
            build-system = with python3Packages; [
              setuptools
            ];
            dependencies = with python3Packages; [
              numpy
              opencv4
            ];
            pyproject = true;
          };
          default = auto-stack;
        });
      devShells = eachSystem (pkgs: with pkgs; {
        default = mkShell {
          packages = [
            pyright
            (treefmtEval pkgs)
          ];
          inputsFrom = [ self.packages.${system}.auto-stack ];
        };
      });
      formatter = eachSystem treefmtEval;
      checks = eachSystem (pkgs: { treefmt = treefmtEval pkgs; });
    };
}
