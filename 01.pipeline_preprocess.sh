if [ "$GENOME_TYPE" = "hg38" ]
then
  GENOME_DIR=/mnt/sfs-genome/GRCh38
  GENOME_NAME=GRCh38
  GENOME_BLACKLIST_NAME=hg38
  BAM_EXT=nodup.bam
  Q30_EXT=nodup.q30.bam
  CORES=30
else
  GENOME_DIR=/mnt/sfs-genome/mm10
  GENOME_NAME=mm10
  GENOME_BLACKLIST_NAME=mm10
  BAM_EXT=mm10.nodup.bam
  Q30_EXT=mm10.nodup.q30.bam
  CORES=30
fi


echo "use ${GENOME_NAME} in ${GENOME_DIR}"

mkdir -p logging contamination tmp

if [ -f "${OUTPUT_DIR}"/"$SAMPLE_NAME".8m."$Q30_EXT" ]
then
  exit 0
fi

if ! [ -f "$SAMPLE_NAME".2.fastq -a -f "$SAMPLE_NAME".1.fastq ]
then
  echo '[Running] decompressing'
  gzip -dkfc "$SOURCE/${SAMPLE_NAME}_1".fq.gz > "$SAMPLE_NAME".1.fq && mv "$SAMPLE_NAME".1.fq "$SAMPLE_NAME".1.fastq &
  gzip -dkfc "$SOURCE/${SAMPLE_NAME}_2".fq.gz > "$SAMPLE_NAME".2.fq && mv "$SAMPLE_NAME".2.fq "$SAMPLE_NAME".2.fastq &
  wait;
else
  echo "[Skip] decompress"
fi

if [ ! -f "$SAMPLE_NAME".2.fastq.clipper ]
then
  echo "[Running] trimmomatic adater sequence"

  java -jar Trimmomatic-0.39/trimmomatic-0.39.jar PE -threads 30 ${SAMPLE_NAME}.1.fastq ${SAMPLE_NAME}.2.fastq "$SAMPLE_NAME.1.tmp.clipper.fastq" "$SAMPLE_NAME.1.tmp.unpaired.clipper.fastq" "$SAMPLE_NAME.2.tmp.clipper.fastq" "$SAMPLE_NAME.2.tmp.unpaired.clipper.fastq" ILLUMINACLIP:Trimmomatic-0.39/adapters/NexteraPE-PE.fa:2:30:10:2:True LEADING:3 TRAILING:3 MINLEN:36 > logging/"$SAMPLE_NAME".trimmomatic.log 2>&1 \
  && mv "$SAMPLE_NAME.1.tmp.clipper.fastq" "$SAMPLE_NAME.1.fastq.clipper" \
  && mv "$SAMPLE_NAME.2.tmp.clipper.fastq" "$SAMPLE_NAME.2.fastq.clipper"

else
  echo "[Skip] trimmomatic"
fi

rm "$SAMPLE_NAME".1.fastq "$SAMPLE_NAME".2.fastq

if [ ! -f "$SAMPLE_NAME".bwa.raw.bam ]
then
  echo "[Running] bwa mem (clean reads mapping)"
  bwa-mem2-2.2.1_x64-linux/bwa-mem2 mem "${GENOME_DIR}/${GENOME_NAME}".fa -t "$CORES" -o "$SAMPLE_NAME".bwa.raw.sam.tmp "$SAMPLE_NAME".1.fastq.clipper "$SAMPLE_NAME".2.fastq.clipper > logging/"$SAMPLE_NAME".bwa.log 2>&1 \
  && mv "$SAMPLE_NAME".bwa.raw.sam.tmp "$SAMPLE_NAME".bwa.raw.sam

  echo 'Start compressing SAM to reduce rsync costs'
  samtools view -Sb "$SAMPLE_NAME".bwa.raw.sam  -o "$SAMPLE_NAME".bwa.raw.bam.tmp  -@ "$CORES"  && mv "$SAMPLE_NAME".bwa.raw.bam.tmp "$SAMPLE_NAME".bwa.raw.bam
else
  echo "[Skip] bwa mem"
fi

if [ ! -f "$SAMPLE_NAME".fixed.bam ]
then
  echo "[Running] samtools fixmate"
  samtools  fixmate -m "$SAMPLE_NAME".bwa.raw.sam  "$SAMPLE_NAME".fixed.tmp.bam -@ "$CORES" \
  && mv "$SAMPLE_NAME".fixed.tmp.bam "$SAMPLE_NAME".fixed.bam
else
  echo "[Skip] samtools fixmate"
fi

if [ ! -f "$SAMPLE_NAME".bam ]
then
  echo "[Running] samtools filter"
  samtools view -b -F 3340 "$SAMPLE_NAME".fixed.bam -@ "$CORES" | samtools sort - -o "$SAMPLE_NAME".bam.tmp -@ "$CORES" \
  && mv "$SAMPLE_NAME".bam.tmp "$SAMPLE_NAME".bam
else
  echo "[Skip] .bam"
fi

if [ ! -f "$SAMPLE_NAME"."$BAM_EXT" ]
then
  echo "[Running] samtools markdup"
  samtools markdup -r "$SAMPLE_NAME".bam "$SAMPLE_NAME".nodup.tmp.bam -m s -f "$SAMPLE_NAME".dup -@ "$CORES" > /dev/null 2>&1 \
  && mv "$SAMPLE_NAME".nodup.tmp.bam "$SAMPLE_NAME"."$BAM_EXT"
else
  echo "[Skip] .nodup.bam"
fi

if [ ! -f "$SAMPLE_NAME"."$Q30_EXT" ]
then
  echo "[Running] samtools q30 filter"
  samtools view "$SAMPLE_NAME"."$BAM_EXT" -b  -q 30 -o "$SAMPLE_NAME".tmp."$Q30_EXT" -@ "$CORES" \
  && mv "$SAMPLE_NAME".tmp."$Q30_EXT" "$SAMPLE_NAME"."$Q30_EXT"
else
  echo "[Skip] .nodup.q30.bam"
fi

if [ ! -f "$SAMPLE_NAME".nodup.q30.bg ]
then
  echo "[Running] bedtools genomecov"
  bedtools genomecov -bga -pc -ibam "$SAMPLE_NAME"."$Q30_EXT" > "$SAMPLE_NAME".nodup.q30.bg
else
  echo "[Skip] bedtools"
fi

if [ ! -f "$SAMPLE_NAME".alignment.log ]
then
  echo '[Running] alignment'
  ALIGNMENT_LOG_FILE="$SAMPLE_NAME".alignment.log
  echo "seqID,Input,bam,ProperBam,nodup.bam,ProperNodupBam,nodup.q30.bam,ProperNodupQ30Bam,rate1,rate2,GenomicCov,dupRate" > "$ALIGNMENT_LOG_FILE.tmp"
  alignment_input=$(( $(wc -l "$SAMPLE_NAME".1.fastq.clipper | cut -d' ' -f1) / 4 ))
  samtools flagstat "$SAMPLE_NAME".bam -@ "$CORES" > "$SAMPLE_NAME".flagstat.tmp
  alignment_bam=$(( $(grep "in total" "$SAMPLE_NAME".flagstat.tmp | cut -d' ' -f1) / 2 ))
  alignment_ProperBam=$(( $(grep "properly paired" "$SAMPLE_NAME".flagstat.tmp | cut -d' ' -f1) / 2 ))

  samtools flagstat "$SAMPLE_NAME"."$BAM_EXT" -@ "$CORES" > "$SAMPLE_NAME".flagstat.tmp
  alignment_nbam=$(( $(grep "in total" "$SAMPLE_NAME".flagstat.tmp | cut -d' ' -f1) / 2 ))
  alignment_nProperBam=$(( $(grep "properly paired" "$SAMPLE_NAME".flagstat.tmp | cut -d' ' -f1) / 2 ))

  samtools flagstat "$SAMPLE_NAME"."$Q30_EXT" -@ "$CORES" > "$SAMPLE_NAME".flagstat.tmp
  alignment_nqbam=$(( $(grep "in total" "$SAMPLE_NAME".flagstat.tmp | cut -d' ' -f1) / 2 ))
  alignment_nqProperBam=$(( $(grep "properly paired" "$SAMPLE_NAME".flagstat.tmp | cut -d' ' -f1) / 2 ))

  rm -f "$SAMPLE_NAME".flagstat.tmp
  alignment_rate1=$( printf "%6f" "$( echo "scale=6;$alignment_bam / $alignment_input" | bc )" )
  alignment_rate2=$( printf "%6f" " $( echo "scale=6;$alignment_nqbam * 1.0 / $alignment_input" | bc )" )

  alignment_cov_number=$( awk '{if ($4 != 0) sum+=$3-$2;}END{printf "%.f", sum}' "$SAMPLE_NAME".nodup.q30.bg )
  alignment_cov_total=$( cut -f2 "$GENOME_DIR"/"$GENOME_NAME".genome | awk  '{sum+=$1}END{printf "%.f", sum}' )
  alignment_cov_rate=$( printf "%f" " $( echo "scale=6;$alignment_cov_number / $alignment_cov_total" | bc )" )

  dupRate=$( printf "%6f" "$( echo "scale=6;($alignment_bam - $alignment_nbam) / $alignment_bam" | bc )" )

  echo "$SAMPLE_NAME,$alignment_input,$alignment_bam,$alignment_ProperBam,$alignment_nbam,$alignment_nProperBam,$alignment_nqbam,$alignment_nqProperBam,$alignment_rate1,$alignment_rate2,$alignment_cov_rate,$dupRate" >> "$ALIGNMENT_LOG_FILE.tmp" \
  && mv "$ALIGNMENT_LOG_FILE.tmp" "$ALIGNMENT_LOG_FILE"
fi

if [ ! -f contamination/"$SAMPLE_NAME".un.100000.out ]
then
  echo "[Running] contamination test"

  ### CONTAMINATION
  samtools view -S -f 4 "$SAMPLE_NAME".bwa.raw.sam -@ "$CORES" | awk '{OFS="\t"; print ">"$1"\n"$10}' > contamination/"$SAMPLE_NAME".un.fa
  head contamination/"$SAMPLE_NAME.un.fa" -n 100000|blastn -query - -db "/DATA/nt" -outfmt '6 staxids sscinames stitle qseqid sseqid sstart send pident length mismatch gapopen qstart qend evalue bitscore qcovs' -evalue 1e-10 -max_target_seqs 1 -out contamination/"$SAMPLE_NAME.un.100000.out" -num_threads "$CORES" > logging/"$SAMPLE_NAME".blastn.log 2>&1
  echo -e $SAMPLE_NAME"\t"$(less contamination/$SAMPLE_NAME.un.100000.out|awk -F "\t" '{a[$3]++}END{for( i in a){print i,a[i] | "sort -nrk 2"}}' |head)"\n" > contamination/$SAMPLE_NAME.out.single
fi

wait
