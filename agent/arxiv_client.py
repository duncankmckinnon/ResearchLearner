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
                          date_from: str = None, categories: List[str] = None) -> ArxivResult:
        """Search for papers on ArXiv"""
        try:
            import arxiv
            
            # Build search query
            search_query = query
            if categories:
                # Add category filters
                category_filter = " OR ".join([f"cat:{cat}" for cat in categories])
                search_query = f"({query}) AND ({category_filter})"
            
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
            
            # For now, return metadata and abstract
            # In a full implementation, you'd extract text from the PDF
            paper_content = {
                "paper_id": paper_id,
                "title": paper.title,
                "authors": [str(author) for author in paper.authors],
                "abstract": paper.summary,
                "content": f"Abstract: {paper.summary}\n\n[Full PDF content extraction would require additional PDF processing libraries]",
                "categories": paper.categories,
                "published": paper.published.isoformat() if paper.published else "",
                "pdf_path": pdf_path if os.path.exists(pdf_path) else None
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
            for paper in papers[:3]:  # Download top 3 papers
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