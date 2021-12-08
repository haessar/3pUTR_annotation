import asyncio
import logging
import os.path

from asgiref.sync import sync_to_async
import gffutils
import pysam

from exceptions import MACS2Error
from utils import cached, consume_lines
from constants import CACHE_DIR, LOG_DIR, PYSAM_STRAND_ARGS


def pysam_strand_split(bam_basename, args, strand):
    if not os.path.isfile(cached(bam_basename + '.%s.bam' % strand)):
        logging.info("Splitting %s strand from %s." % (strand, args.BAM_IN))
        pysam.view("--threads", "6", "-b", *PYSAM_STRAND_ARGS[strand], "-o", cached(bam_basename + '.%s.bam' % strand), args.BAM_IN, catch_stdout=False)
        logging.info("Finished splitting %s strand." % strand)


async def create_db(gff_in):
    gff_db = cached(os.path.basename(os.path.splitext(gff_in)[0] + '.db'))
    if not os.path.isfile(gff_db):
        logging.info('Creating gff db.')
        await sync_to_async(gffutils.create_db)(gff_in, gff_db, force=True)
        logging.info('Finished creating gff db.')
    return gffutils.FeatureDB(gff_db)


async def call_peaks(bam_basename, strand):
    """
    Call MACS2 asynchronously for forward and reverse strand BAM files.
    """
    if not os.path.isfile(cached("%s_peaks.broadPeak" % strand)):
        logging.info("Calling peaks for %s strand with MACS2." % strand)
        process = await asyncio.create_subprocess_exec(
            "macs2", "callpeak", "-t", cached(bam_basename + '.%s.bam' % strand), "-n", strand, "--nomodel", "--extsize", "100", "--broad", "--outdir", CACHE_DIR,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT
        )
        stdout_task = asyncio.create_task(consume_lines(process.stdout, os.path.join(LOG_DIR, "%s_macs2.log" % strand)))
        await stdout_task
        exit_code = await process.wait()
        if exit_code != 0:
            logging.error("MACS2 returned an error.")
            raise MACS2Error("Check %s_macs2.log." % strand)
        logging.info("Finished calling %s strand peaks." % strand)
    else:
        logging.info("Using cached %s strand peaks file." % strand)