# Guide to updating the Nesta Brain knowledge base

## Introduction

This document is a guide to updating the knowledge base which Nesta Brain draws from, that is, how to ingest new information into the LanceDB vector database used for retrieval.

Currently, the code is set up to ingest web pages and online PDFs.

The code reflects the data ingestion needs for early prototypes of the Nesta Brain project, written fairly rapidly, and may need to be reviewed and simplified for optimal future maintainability.


## Relevant code

Most of the relevant code is in retrieval/db/ingest.py. See also scraping/scrape.py and scraping/scrape_pdf.py.  

## Settings

A list of settings and options for ingesting text sources is given at the top of \__main__ in retrieval/db/ingest.py.

**mode**: Literal["web_dump","web_search"]. If the value is "web_search" then webpages resulting from a Google programmable search will be ingested (see "settings relevant to web_search mode"). If the value is "web_dump" then data which has already been downloaded from the Nesta website and is contained in a directory with path WEBSITE_DATA_PATH will be ingested.
