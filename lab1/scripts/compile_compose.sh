#!/usr/bin/env bash
TRANSDUCER=${1}
ACCEPTOR=${2}

fstcompile -isymbols=vocab/chars.syms -osymbols=vocab/chars.syms fsts/${TRANSDUCER}.fst fsts/${TRANSDUCER}.binfst
fstcompile -isymbols=vocab/chars.syms -osymbols=vocab/words.syms fsts/${ACCEPTOR}.fst fsts/${ACCEPTOR}.binfst
fstrmepsilon fsts/${ACCEPTOR}.binfst | fstdeterminize | fstminimize > ${ACCEPTOR}.binfst
fstcompile -isymbols=vocab/chars.syms -osymbols=vocab/words.syms fsts/${ACCEPTOR}.fst fsts/${ACCEPTOR}.binfst
fstarcsort --sort_type=olabel fsts/${TRANSDUCER}.binfst fsts/${TRANSDUCER}_sorted.binfst
fstarcsort --sort_type=ilabel fsts/${ACCEPTOR}.binfst fsts/${ACCEPTOR}_sorted.binfst
fstcompose fsts/${TRANSDUCER}_sorted.binfst fsts/${ACCEPTOR}_sorted.binfst fsts/${TRANSDUCER}${ACCEPTOR}.binfst
