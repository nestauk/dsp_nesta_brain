# Guide to updating the Nesta Brain knowledge base

## Introduction

This document is a guide to updating the knowledge base which Nesta Brain draws from, that is, how to ingest new information into the LanceDB vector database used for retrieval.

Currently, the code is set up to ingest web pages and online PDFs.

The code reflects the data ingestion needs for early prototypes of the Nesta Brain project, written fairly rapidly, and may need to be reviewed and simplified for optimal future maintainability.


## Relevant code

Most of the relevant code is in `retrieval/db/ingest.py`. See also `scraping/scrape.py` and `scraping/scrape_pdf.py`.  

## Data

[MENTION ABOUT the DATA DUMP HERE – get from S3]

## Settings

A list of settings and options for ingesting text sources is given at the top of \__main__ in retrieval/db/ingest.py.

**`mode`**: `Literal["web_dump","web_search"]` 
> If the value is `"web_search"` then webpages resulting from a Google programmable search will be ingested (see "settings relevant to web_search mode"). If the value is `"web_dump"` then data which has already been downloaded from the Nesta website and is contained in a directory with path `WEBSITE_DATA_PATH` will be ingested. 

**`replace`**: `bool`
> if `True`, if a document already exists in the DB, any chunks previously derived from it will be deleted and replaced; if `False`, previous chunks will be left but new chunks will not be added. This is to avoid duplication of chunks.

*Settings relevant to `web_dump` mode*

**`pdf_mode`**:
> if `True`, the system will look for PDFs downloaded from the website and stored in the directory with path `PDF_PATH`; if `False` it will look for downloaded webpages contained in `WEBSITE_DATA_PATH`

**`download_button_pdf_only`**:
> One problem we encountered while ingesting PDFs is detecting and inserting their metadata (tile, publication date). We found this could not be reliably autmatically identified from the PDF itself. For this reason, where a webpage has obviously been created to promote a new report, it is simplest to assume the PDF has the same metadata as the webpage. If `download_button_pdf_only` is True, the the system will only ingest PDFs linked to by a red 'Download' button on the originating webpage; `False`, it will look for and ingest all PDFs linked to by the originating webapge, as long as they are on the Nesta website. Note that if `cautious` (see below) is False, then all PDFs are given the same title and publication date as the originating webpage, regardless of whether `download_button_pdf_only` is `True` or `False`. Note that this could mean a significant nuber of PDFs in the database with inaccurate metadata.
