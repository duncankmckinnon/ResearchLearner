import asyncio
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import os

logger = logging.getLogger("simple_arxiv_client")

@dataclass
class ArxivResult:
    """Result from ArXiv operations"""
    success: bool
    data: Any
    error: Optional[str] = None

class SimpleArxivClient:
    """Simple ArXiv client using the arxiv Python library"""
    
    def __init__(self):
        self.storage_path = os.path.expanduser("~/.arxiv-mcp-server/papers")
        os.makedirs(self.storage_path, exist_ok=True)
        
    async def search_papers(self, query: str, max_results: int = 10, 
                          date_from: Optional[str] = None, categories: Optional[List[str]] = None) -> ArxivResult:
        """Search for papers on ArXiv"""
        try:
            import arxiv
            
            # Build search query
            search_query = query
            if categories:
                # Add category filters
                category_filter = " OR ".join([f"cat:{cat}" for cat in categories])
                search_query = f"({query}) AND ({category_filter}) {f'AND publishedDate:[{date_from} TO *]' if date_from else ''}"
            
            # Create search
            search = arxiv.Search(
                query=search_query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            # Create client and execute search
            client = arxiv.Client()
            papers = []
            for result in client.results(search):
                paper_data = {
                    "id": result.entry_id.split("/")[-1],  # Extract arXiv ID
                    "title": result.title,
                    "authors": [str(author) for author in result.authors],
                    "abstract": result.summary,
                    "categories": result.categories,
                    "published": result.published.isoformat() if result.published else "",
                    "pdf_url": result.pdf_url,
                    "entry_id": result.entry_id
                }
                papers.append(paper_data)
            
            return ArxivResult(
                success=True,
                data={
                    "papers": papers,
                    "papers_found": len(papers)
                }
            )
            
        except ImportError:
            return ArxivResult(
                success=False,
                data=None,
                error="arxiv library not installed. Install with: pip install arxiv"
            )
        except Exception as e:
            logger.error(f"Error searching ArXiv: {str(e)}")
            return ArxivResult(success=False, data=None, error=str(e))
    
    async def download_paper(self, paper_id: str) -> ArxivResult:
        """Download a paper by ArXiv ID"""
        try:
            import arxiv
            
            # Search for the specific paper
            search = arxiv.Search(id_list=[paper_id])
            client = arxiv.Client()
            paper = next(client.results(search), None)
            
            if not paper:
                return ArxivResult(
                    success=False,
                    data=None,
                    error=f"Paper {paper_id} not found"
                )
            
            # Download PDF
            pdf_path = os.path.join(self.storage_path, f"{paper_id}.pdf")
            paper.download_pdf(dirpath=self.storage_path, filename=f"{paper_id}.pdf")
            
            return ArxivResult(
                success=True,
                data={
                    "paper_id": paper_id,
                    "pdf_path": pdf_path,
                    "title": paper.title,
                    "downloaded": True
                }
            )
            
        except ImportError:
            return ArxivResult(
                success=False,
                data=None,
                error="arxiv library not installed. Install with: pip install arxiv"
            )
        except Exception as e:
            logger.error(f"Error downloading paper {paper_id}: {str(e)}")
            return ArxivResult(success=False, data=None, error=str(e))
    
    async def read_paper(self, paper_id: str) -> ArxivResult:
        """Read the content of a downloaded paper"""
        try:
            import arxiv

            # First get the paper metadata
            search = arxiv.Search(id_list=[paper_id])
            client = arxiv.Client()
            paper = next(client.results(search), None)

            if not paper:
                return ArxivResult(
                    success=False,
                    data=None,
                    error=f"Paper {paper_id} not found"
                )

            # Check if PDF exists
            pdf_path = os.path.join(self.storage_path, f"{paper_id}.pdf")

            # Extract full text from PDF if it exists
            full_content = ""
            pdf_metadata = {}
            if os.path.exists(pdf_path):
                try:
                    full_content, pdf_metadata = self._extract_pdf_text(pdf_path)
                    logger.info(f"Extracted {len(full_content)} characters from PDF {paper_id}")
                except Exception as e:
                    logger.warning(f"Failed to extract PDF text for {paper_id}: {str(e)}")
                    full_content = f"[PDF text extraction failed: {str(e)}]"

            # Combine abstract and full content
            if full_content and not full_content.startswith("[PDF text extraction failed"):
                content = f"Abstract: {paper.summary}\n\nFull Paper Content:\n{full_content}"
            else:
                content = f"Abstract: {paper.summary}\n\n[PDF not available or text extraction failed]"

            paper_content = {
                "paper_id": paper_id,
                "title": paper.title,
                "authors": [str(author) for author in paper.authors],
                "abstract": paper.summary,
                "content": content,
                "categories": paper.categories,
                "published": paper.published.isoformat() if paper.published else "",
                "pdf_path": pdf_path if os.path.exists(pdf_path) else None,
                "pdf_metadata": pdf_metadata
            }

            return ArxivResult(success=True, data=paper_content)
            
        except ImportError:
            return ArxivResult(
                success=False,
                data=None,
                error="arxiv library not installed. Install with: pip install arxiv"
            )
        except Exception as e:
            logger.error(f"Error reading paper {paper_id}: {str(e)}")
            return ArxivResult(success=False, data=None, error=str(e))

    def _extract_pdf_text(self, pdf_path: str) -> tuple[str, dict]:
        """Extract text from a PDF file using PyMuPDF and return text + metadata"""
        import fitz  # PyMuPDF
        import re

        # Open the PDF
        doc = fitz.open(pdf_path)
        text = ""

        # Extract text from each page
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text()
            text += "\n\n"  # Add page separator

        doc.close()

        # Extract URLs and DOIs before cleaning
        urls = re.findall(r'https?://[^\s]+', text)
        dois = re.findall(r'doi:[^\s]+', text)

        # Clean up the text
        text = self._clean_extracted_text(text)

        metadata = {
            "urls": list(set(urls)),  # Remove duplicates
            "dois": list(set(dois))   # Remove duplicates
        }

        return text, metadata

    def _clean_extracted_text(self, text: str) -> str:
        """Clean and format extracted text"""
        import re

        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)

        # Remove page headers/footers (common patterns)
        text = re.sub(r'\n\d+\n', '\n', text)  # Page numbers on their own line
        text = re.sub(r'arXiv:\d+\.\d+v\d+.*?\n', '', text)  # ArXiv headers

        # Remove URLs and DOIs from text (we saved them in metadata)
        text = re.sub(r'https?://[^\s]+', '', text)
        text = re.sub(r'doi:[^\s]+', '', text)

        return text.strip()

    async def list_papers(self) -> ArxivResult:
        """List all downloaded papers"""
        try:
            papers = []
            if os.path.exists(self.storage_path):
                for filename in os.listdir(self.storage_path):
                    if filename.endswith('.pdf'):
                        paper_id = filename.replace('.pdf', '')
                        papers.append({
                            "paper_id": paper_id,
                            "filename": filename,
                            "path": os.path.join(self.storage_path, filename)
                        })
            
            return ArxivResult(
                success=True,
                data={"papers": papers, "count": len(papers)}
            )
            
        except Exception as e:
            logger.error(f"Error listing papers: {str(e)}")
            return ArxivResult(success=False, data=None, error=str(e))

class SimpleResearchAgent:
    """Research agent using simple ArXiv client"""
    
    def __init__(self):
        self.arxiv_client = SimpleArxivClient()
        
    async def research_topic(self, topic: str, max_papers: int = 10) -> Dict[str, Any]:
        """Research a topic using ArXiv papers"""
        try:
            # Search for papers
            search_result = await self.arxiv_client.search_papers(topic, max_results=max_papers)
            
            if not search_result.success:
                return {
                    "topic": topic,
                    "error": search_result.error,
                    "papers": []
                }
            
            papers = search_result.data.get("papers", [])
            
            # Download top papers
            downloaded_papers = []
            for paper in papers:  # Download all papers
                paper_id = paper.get("id", "")
                if paper_id:
                    download_result = await self.arxiv_client.download_paper(paper_id)
                    if download_result.success:
                        downloaded_papers.append(paper_id)
            
            return {
                "topic": topic,
                "papers_found": len(papers),
                "papers_downloaded": len(downloaded_papers),
                "papers": papers,
                "downloaded_paper_ids": downloaded_papers
            }
            
        except Exception as e:
            logger.error(f"Error researching topic {topic}: {str(e)}")
            return {
                "topic": topic,
                "error": str(e),
                "papers": []
            }
    
    async def analyze_paper(self, paper_id: str) -> Dict[str, Any]:
        """Analyze a specific paper"""
        try:
            # First download the paper if not already downloaded
            download_result = await self.arxiv_client.download_paper(paper_id)
            if not download_result.success:
                return {
                    "paper_id": paper_id,
                    "error": f"Failed to download paper: {download_result.error}"
                }
            
            # Read the paper content
            read_result = await self.arxiv_client.read_paper(paper_id)
            if not read_result.success:
                return {
                    "paper_id": paper_id,
                    "error": f"Failed to read paper: {read_result.error}"
                }
            
            return read_result.data
            
        except Exception as e:
            logger.error(f"Error analyzing paper {paper_id}: {str(e)}")
            return {
                "paper_id": paper_id,
                "error": str(e)
            }