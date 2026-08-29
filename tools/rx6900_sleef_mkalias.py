#!/usr/bin/env python3
"""Generate SLEEF alias headers.

Windows Device Guard blocks the unsigned mkalias.exe host tool. This script
matches third_party/sleef/src/libm/mkalias.c so CMake can run it via Python.
Place it next to funcproto.h (the build script copies it into the sleef tree).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


def load_func_list(path: Path) -> list[tuple[str, int, int, int]]:
    text = path.read_text(encoding="utf-8")
    funcs: list[tuple[str, int, int, int]] = []
    for match in re.finditer(
        r'\{\s*"(?P<name>[^"]+)"\s*,\s*(?P<ulp>-?\d+)\s*,\s*(?P<ulp_suffix>\d+)\s*,\s*(?P<func_type>\d+)\s*,\s*(?P<flags>\d+)\s*\}',
        text,
    ):
        funcs.append(
            (
                match.group("name"),
                int(match.group("ulp")),
                int(match.group("func_type")),
                int(match.group("flags")),
            )
        )
    return funcs


def main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[1] == "0":
        return 0
    if len(argv) < 6:
        sys.stderr.write(
            f"Usage : {argv[0]} <vector width> <vector FP type> <vector int type> <mangled ISA> <extension>\n"
        )
        return 1

    vw = int(argv[1])
    fptype = 0 if vw >= 0 else 1
    vw = abs(vw)
    mangledisa = argv[4]
    gen_alias_vector_abi = mangledisa[0] != "-"
    isaname = argv[5] if len(argv) == 6 else ""
    vectorcc = ""

    arg_type2 = ["a0", "a0, a1", "a0", "a0, a1", "a0", "a0, a1, a2", "a0", "a0", "a0"]
    type_spec_s = ["", "f"]
    type_spec = ["d", "f"]
    vparameter_str = ["v", "vv", "", "vv", "v", "vvv", "", "", ""]

    fp_vec = argv[2]
    int_vec = argv[3]
    pair = "vfloat2" if fptype else "vdouble2"
    return_type = [fp_vec, fp_vec, pair, fp_vec, int_vec, fp_vec, pair, "int", "void *"]
    arg_type0 = [
        fp_vec,
        f"{fp_vec}, {fp_vec}",
        fp_vec,
        f"{fp_vec}, {int_vec}",
        fp_vec,
        f"{fp_vec}, {fp_vec}, {fp_vec}",
        fp_vec,
        "int",
        "int",
    ]
    arg_type1 = [
        f"{fp_vec} a0",
        f"{fp_vec} a0, {fp_vec} a1",
        f"{fp_vec} a0",
        f"{fp_vec} a0, {int_vec} a1",
        f"{fp_vec} a0",
        f"{fp_vec} a0, {fp_vec} a1, {fp_vec} a2",
        f"{fp_vec} a0",
        "int a0",
        "int a0",
    ]

    funcs = load_func_list(Path(__file__).with_name("funcproto.h"))
    guard = "__SLEEFSIMDSP_C__" if fptype else "__SLEEFSIMDDP_C__"
    out = [f"#ifdef {guard}\n", "#ifdef ENABLE_ALIAS\n"]

    if len(argv) == 6:
        for name, ulp, func_type, flags in funcs:
            if fptype == 0 and (flags & 2) != 0:
                continue
            if ulp >= 0:
                out.append(
                    f"EXPORT CONST {return_type[func_type]} Sleef_{name}{type_spec[fptype]}{vw}_u{ulp:02d}({arg_type0[func_type]}) __attribute__((alias(\"Sleef_{name}{type_spec[fptype]}{vw}_u{ulp:02d}{isaname}\"))) {vectorcc};\n"
                )
                if gen_alias_vector_abi and vparameter_str[func_type]:
                    out.append(
                        f"EXPORT CONST VECTOR_CC {return_type[func_type]} _ZGV{mangledisa}N{vw}{vparameter_str[func_type]}_Sleef_{name}{type_spec_s[fptype]}_u{ulp:02d}({arg_type0[func_type]}) __attribute__((alias(\"Sleef_{name}{type_spec[fptype]}{vw}_u{ulp:02d}{isaname}\"))){vectorcc};\n"
                    )
            else:
                out.append(
                    f"EXPORT CONST {return_type[func_type]} Sleef_{name}{type_spec[fptype]}{vw}({arg_type0[func_type]}) __attribute__((alias(\"Sleef_{name}{type_spec[fptype]}{vw}_{isaname}\"))) {vectorcc};\n"
                )
                if gen_alias_vector_abi and vparameter_str[func_type]:
                    out.append(
                        f"EXPORT CONST VECTOR_CC {return_type[func_type]} _ZGV{mangledisa}N{vw}{vparameter_str[func_type]}_Sleef_{name}{type_spec_s[fptype]}({arg_type0[func_type]}) __attribute__((alias(\"Sleef_{name}{type_spec[fptype]}{vw}_{isaname}\"))){vectorcc};\n"
                    )
        out.append("\n")

    out.append("#else // #ifdef ENABLE_ALIAS\n")
    if len(argv) == 6:
        for name, ulp, func_type, flags in funcs:
            if fptype == 0 and (flags & 2) != 0:
                continue
            if ulp >= 0:
                out.append(
                    f"EXPORT CONST {return_type[func_type]} {vectorcc} Sleef_{name}{type_spec[fptype]}{vw}_u{ulp:02d}({arg_type1[func_type]}) {{ return Sleef_{name}{type_spec[fptype]}{vw}_u{ulp:02d}{isaname}({arg_type2[func_type]}); }}\n"
                )
            else:
                out.append(
                    f"EXPORT CONST {return_type[func_type]} {vectorcc} Sleef_{name}{type_spec[fptype]}{vw}({arg_type1[func_type]}) {{ return Sleef_{name}{type_spec[fptype]}{vw}_{isaname}({arg_type2[func_type]}); }}\n"
                )
        out.append("\n")

    out.append("#endif // #ifdef ENABLE_ALIAS\n")
    out.append(f"#endif // #ifdef {guard}\n")
    sys.stdout.write("".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
