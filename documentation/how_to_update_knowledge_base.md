# Guide to updating the Nesta Brain knowledge base

[work in progress]

**NB some of the instructions below will need branch `ingest-corrections` to be merged and are not correct in `dev` as of 04/12/24**

## Introduction

This document is a guide to updating the knowledge base of documents which Nesta Brain uses to answer questions; that is, how to ingest new information into the LanceDB vector database used for retrieval.

Currently, the code is set up to ingest web pages (from any site) and PDFs from the Nesta website.

The code reflects the data ingestion needs for early prototypes of the Nesta Brain project, written and deployed fairly rapidly, and with limited time for tidying and rationalisation. It may need to be reviewed and simplified for optimal future maintainability, as well as extended to allow ingestion from other sources.


## Relevant code

Most of the relevant code is in `retrieval/db/ingest.py`. See also `scraping/scrape.py` and `scraping/scrape_pdf.py` for web and PDF scraping functions.  

## Embeddings model

OpenAI's `text-embedding-3-small` model was used for the database versions called `full_site_demo_db` and `full_site_demo_db_with_pdfs`. Future users may want to experiment with different embeddings models. The embeddings model can be set in `retrieval/db/ingest.py` via `MODEL_NAME`. Note that if the embeddings model is changed in `MODEL_NAME` then: (i) the variables encapsulating the rate limits, `RPM_RATE_LIMIT` and `TPM_RATE_LIMIT`, may also need to be changed; and (ii) the embeddings model used by the retriever in `retrieval/retrieve.py` should also be changed [**to do – this should be set in `config.py`**].

## Data

[MENTION ABOUT the DATA DUMP HERE – get from S3]

In `web_dump` mode (see below) the details of the webpages to scrape are taken from the metadata file with path `METADATA_PATH` and read into a dataframe. with any webpages with a `_status_code` of 200 removed from the dataframe. The dataframe should only contain webpages which have been successfully downloaded.

## PDF scraping

[details]

## Settings

A list of settings and options for ingesting text sources is given at the top of `__main__` in retrieval/db/ingest.py.

**`mode`**: `Literal["web_dump","web_search"]` 
> If the value is `"web_search"` then webpages resulting from a Google programmable search will be ingested (see "settings relevant to web_search mode"). If the value is `"web_dump"` then data which has already been downloaded from the Nesta website and is contained in a directory with path `WEBSITE_DATA_PATH` will be ingested. 

**`replace`**: `bool`
> if `True`, if a document already exists in the DB, any chunks previously derived from it will be deleted and replaced; if `False`, previous chunks will be left but new chunks will not be added. This is to avoid duplication of chunks.

*Settings relevant to `web_dump` mode*

**`pdf_mode`**: `bool`
> if `True`, the system will look for PDFs downloaded from the website and stored in the directory with path `PDF_PATH`; if `False` it will look for downloaded webpages contained in `WEBSITE_DATA_PATH`

**`download_button_pdf_only`**: `bool`
> One problem we encountered while ingesting PDFs is detecting and inserting their metadata (tile, publication date). We found this could not be reliably autmatically identified from the PDF itself. For this reason, where a webpage has obviously been created to promote a new report, it is simplest to assume the PDF has the same metadata as the webpage. If `download_button_pdf_only` is True, the the system will only ingest PDFs linked to by a red 'Download' button on the originating webpage; if `False`, it will look for and ingest all PDFs linked to by the originating webapge, as long as they are on the Nesta website. Note that if `cautious` (see below) is False, then all PDFs are given the same title and publication date as the originating webpage, regardless of whether `download_button_pdf_only` is `True` or `False`. Note that this could mean a significant nuber of PDFs in the database with inaccurate metadata.

**`cautious`**: `bool`
> if `True` the system will ask whether you want to scrape each PDF in turn and will ask whether metadata guesses are correct; if `False`, it will scrape each PDF automatically and assume the metadata from the originating webpage, as described above. Setting `cautious` to True is very slow and intended for testing.

**`start_index`**: `int`
> the row of the metadata dataframe (see **Data** section) to start ingesting from, with the first row at index `0`. `start_index` is taken from the first command line argument, or defaults to `0`.

**`batch_size`**: `int`
> the number of webpages or PDFs to get embeddings for and ingest at a time. Note that if `batch_size` is too high then you will get error messages back from OpenAI (see **Known issues**). Users are encouraged to experiment with `batch_size`. PDFs can be large and slow to scrape, so a very low batch_size (<5) is recommended if `pdf_mode` is `True`. A `batch_size` of 50 for webpages and 1 for PDFs was used when the DB was originally set up. Batch sizes > 100 for webapges seemed to cause problems.

*Settings relevant to web_search mode*

**`query`**: `str`
> the web search query (as if doing a Google search)

**`site_url`**: `str`
> the url of the website which is the target of the search

**`subdirectories`**:`Optional[List[str]]`
> a list of subdirectories on the website which you wish to limit the search to; note that a separate search will be conducted for each of these subdirectories in turn

*Google Programmable Search limits*: Note that only a 100 search results can be returned from Google Programmable Search for each distinct search, where a distinct search is a combination of query, url and subdirectory. Furthermore, if more than 100 searches a day are required, a billing account will need to be set up. See [this Google webpage](https://developers.google.com/custom-search/v1/overview#:~:text=Custom%20Search%20JSON%20API%20provides,to%2010k%20queries%20per%20day.) for more details.

## Known issues

1. An attempt was made to write code to throttle OpenAI requests and ensure they are kept within rate limits, but this may not have been fully successful. In addition, when `batch_size` is 
There wasn't time to test and troubleshoot the throttle, and was also getting error messages back from OpenAI when we couldn't have been exceeding the rate limits.
t work. Also getting error messages back when we can't possibly be exceeding. 
2. There were some PDFs which didn't scrape successfully and which threw error messages, probably due to size. There wasn't time to investigate and fix this. You may encounter. Can always set start_index following the problem PDF.
