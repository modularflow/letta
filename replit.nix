{pkgs}: {
  deps = [
    pkgs.rustc
    pkgs.libiconv
    pkgs.cargo
    pkgs.libyaml
    pkgs.bash
    pkgs.openssl
    pkgs.libxcrypt
    pkgs.postgresql
  ];
}
