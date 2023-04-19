SPELL_CHECKER=${1}
W=${2}

fstcompile -isymbols=vocab/words.syms -osymbols=vocab/words.syms fsts/${W}.fst fsts/${W}.binfst
fstrmepsilon fsts/${W}.binfst | fstdeterminize | fstminimize > fsts/${W}_opt.binfst
fstarcsort --sort_type=olabel fsts/${SPELL_CHECKER}.binfst fsts/${SPELL_CHECKER}_sorted.binfst
fstarcsort --sort_type=ilabel fsts/${W}_opt.binfst fsts/${W}_sorted.binfst
fstcompose fsts/${SPELL_CHECKER}_sorted.binfst fsts/${W}_sorted.binfst fsts/${SPELL_CHECKER}${W}.binfst
