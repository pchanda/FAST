#!/bin/bash
# This script will run FAST in mode=genotype with linear model.
# The input is from files in DATA.geno
# The output is stored in OUT

for CHR in {1..2}
do
  time ../FAST \
       --tped-file ./DATA.geno/chr$CHR.tped.gz \
       --snpinfo-file ./DATA.geno/chr$CHR.snp.info.gz \
       --mlinfo-file ./DATA.geno/chr$CHR.mlinfo.gz \
       --indiv-file ./DATA.geno/id.txt.gz \
       --out-file ./OUT/output1 \
       --trait-file ./DATA.geno/CC.tfam.gz \
       --gene-set ./DATA.geno/genes12.txt.gz \
       --chr $CHR \
       --maf-cutoff 0.01 \
       --mode genotype \
       --max-perm 100 \
       --linear-bf-perm \
       --linear-minsnp-perm \
       --linear-minsnp-gene-perm \
       --linear-vegas-perm \
       --linear-gwis-perm \
       --linear-gates \
       --flank 5 \
       --n-threads 1
done  

