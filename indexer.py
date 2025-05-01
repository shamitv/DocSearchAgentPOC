import concurrent.futures
from itertools import islice

        "bulk_submission": 0,
        "total_processing": 0
    }
    
    logger.info(f"Opening dump file: {file_path}")
    try:
        file_open_start = time.time()
        with bz2.open(file_path, "rb") as file:
            dump = mwxml.Dump.from_file(file)
            timings["file_opening"] = time.time() - file_open_start
            
            logger.info(f"File opening completed in {timings['file_opening']:.2f}s")
            
            # Process pages in parallel with a maximum of 20 workers
            with concurrent.futures.ProcessPoolExecutor(max_workers=20) as executor:
                # Process the dump in batches to avoid memory issues
                batch_size = 100  # Process pages in batches of 100
                page_count = 0
