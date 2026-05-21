"""
RAG Evaluation Framework
Provides metrics and evaluation tools for RAG system performance assessment.
"""

import logging
import json
import time
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class RAGMetrics:
    """RAG performance metrics container"""
    query: str
    response_time: float
    num_chunks_retrieved: int
    num_chunks_used: int
    context_relevance_score: float = 0.0
    answer_relevance_score: float = 0.0
    answer_accuracy_score: float = 0.0
    hallucination_score: float = 0.0
    citation_accuracy: float = 0.0
    language_consistency: float = 0.0
    retrieval_f1_score: float = 0.0
    response_completeness: float = 0.0
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        if self.metadata:
            data['metadata'] = self.metadata
        return data

class RAGEvaluator:
    """
    Comprehensive RAG evaluation framework with multiple metrics.
    """

    def __init__(self, log_file: str = "rag_metrics.log"):
        self.log_file = log_file
        self.metrics_history: List[RAGMetrics] = []
        logger.info("RAG Evaluator initialized")

    def evaluate_query(self, query: str, retrieved_chunks: List[Dict],
                       generated_answer: str, ground_truth: Optional[str] = None,
                       response_time: float = 0.0) -> RAGMetrics:
        """
        Comprehensive evaluation of a RAG query response.

        Args:
            query: Original user query
            retrieved_chunks: List of retrieved document chunks
            generated_answer: LLM-generated answer
            ground_truth: Expected correct answer (optional)
            response_time: Time taken to generate response

        Returns:
            RAGMetrics object with all evaluation scores
        """
        try:
            # Basic metrics
            num_chunks_retrieved = len(retrieved_chunks)
            num_chunks_used = self._count_chunks_used(generated_answer, retrieved_chunks)

            # Extract content from chunks for analysis
            chunk_contents = [chunk.get('page_content', '') for chunk in retrieved_chunks]

            # Calculate metrics
            context_relevance = self._calculate_context_relevance(query, chunk_contents)
            answer_relevance = self._calculate_answer_relevance(query, generated_answer)
            answer_accuracy = self._calculate_answer_accuracy(generated_answer, ground_truth) if ground_truth else 0.0
            hallucination_score = self._calculate_hallucination_score(generated_answer, chunk_contents)
            citation_accuracy = self._calculate_citation_accuracy(generated_answer, retrieved_chunks)
            language_consistency = self._calculate_language_consistency(generated_answer, chunk_contents)
            retrieval_f1 = self._calculate_retrieval_f1(query, retrieved_chunks)
            completeness = self._calculate_response_completeness(query, generated_answer, chunk_contents)

            # Create metrics object
            metrics = RAGMetrics(
                query=query,
                response_time=response_time,
                num_chunks_retrieved=num_chunks_retrieved,
                num_chunks_used=num_chunks_used,
                context_relevance_score=context_relevance,
                answer_relevance_score=answer_relevance,
                answer_accuracy_score=answer_accuracy,
                hallucination_score=hallucination_score,
                citation_accuracy=citation_accuracy,
                language_consistency=language_consistency,
                retrieval_f1_score=retrieval_f1,
                response_completeness=completeness,
                metadata={
                    'timestamp': datetime.now().isoformat(),
                    'query_length': len(query),
                    'answer_length': len(generated_answer),
                    'chunk_avg_length': np.mean([len(c) for c in chunk_contents]) if chunk_contents else 0,
                }
            )

            # Store in history
            self.metrics_history.append(metrics)

            # Log metrics
            self._log_metrics(metrics)

            return metrics

        except Exception as e:
            logger.error(f"Error evaluating query: {e}")
            # Return basic metrics on error
            return RAGMetrics(
                query=query,
                response_time=response_time,
                num_chunks_retrieved=len(retrieved_chunks),
                num_chunks_used=0,
                metadata={'error': str(e)}
            )

    def _count_chunks_used(self, answer: str, chunks: List[Dict]) -> int:
        """Count how many chunks were actually used in the answer."""
        try:
            used_chunks = 0
            answer_lower = answer.lower()

            for chunk in chunks:
                content = chunk.get('page_content', '').lower()
                # Simple heuristic: check for significant overlap
                words_overlap = len(set(content.split()) & set(answer_lower.split()))
                if words_overlap > 5:  # At least 5 overlapping words
                    used_chunks += 1

            return min(used_chunks, len(chunks))  # Cap at total chunks

        except Exception as e:
            logger.warning(f"Error counting chunks used: {e}")
            return 0

    def _calculate_context_relevance(self, query: str, chunk_contents: List[str]) -> float:
        """Calculate how relevant the retrieved context is to the query."""
        try:
            if not chunk_contents:
                return 0.0

            query_words = set(query.lower().split())
            total_relevance = 0.0

            for content in chunk_contents:
                content_words = set(content.lower().split())
                overlap = len(query_words & content_words)
                relevance = overlap / len(query_words) if query_words else 0.0
                total_relevance += relevance

            return min(total_relevance / len(chunk_contents), 1.0)

        except Exception as e:
            logger.warning(f"Error calculating context relevance: {e}")
            return 0.0

    def _calculate_answer_relevance(self, query: str, answer: str) -> float:
        """Calculate how well the answer addresses the query."""
        try:
            # Simple heuristic based on query term overlap in answer
            query_words = set(query.lower().split())
            answer_words = set(answer.lower().split())

            overlap = len(query_words & answer_words)
            relevance = overlap / len(query_words) if query_words else 0.0

            # Boost if answer contains key question words
            question_words = {'what', 'how', 'when', 'where', 'why', 'who', 'which'}
            if any(word in answer.lower() for word in question_words):
                relevance += 0.1

            return min(relevance, 1.0)

        except Exception as e:
            logger.warning(f"Error calculating answer relevance: {e}")
            return 0.0

    def _calculate_answer_accuracy(self, answer: str, ground_truth: str) -> float:
        """Calculate answer accuracy against ground truth."""
        try:
            if not ground_truth:
                return 0.0

            # Simple semantic similarity (can be enhanced with embeddings)
            answer_words = set(answer.lower().split())
            truth_words = set(ground_truth.lower().split())

            overlap = len(answer_words & truth_words)
            union = len(answer_words | truth_words)

            return overlap / union if union > 0 else 0.0

        except Exception as e:
            logger.warning(f"Error calculating answer accuracy: {e}")
            return 0.0

    def _calculate_hallucination_score(self, answer: str, chunk_contents: List[str]) -> float:
        """Calculate hallucination score (lower is better, 0 = no hallucination)."""
        try:
            if not chunk_contents:
                return 1.0  # High hallucination if no context

            answer_sentences = answer.split('.')
            hallucinated_sentences = 0

            for sentence in answer_sentences:
                sentence = sentence.strip()
                if len(sentence) < 10:  # Skip very short sentences
                    continue

                # Check if sentence content can be found in any chunk
                found_in_context = False
                for chunk in chunk_contents:
                    if sentence.lower() in chunk.lower():
                        found_in_context = True
                        break

                if not found_in_context:
                    hallucinated_sentences += 1

            hallucination_rate = hallucinated_sentences / len([s for s in answer_sentences if len(s.strip()) >= 10])
            return min(hallucination_rate, 1.0)

        except Exception as e:
            logger.warning(f"Error calculating hallucination score: {e}")
            return 0.5

    def _calculate_citation_accuracy(self, answer: str, chunks: List[Dict]) -> float:
        """Calculate accuracy of citations in the answer."""
        try:
            import re

            # Find all citations in answer (Source: Page X)
            citations = re.findall(r'\(Source: Page (\d+)\)', answer)
            cited_pages = set(int(page) for page in citations)

            # Get actual available pages
            available_pages = set()
            for chunk in chunks:
                page = chunk.get('page') or chunk.get('page_number')
                if page:
                    available_pages.add(int(page))

            if not citations:
                return 0.5  # Neutral score if no citations

            if not available_pages:
                return 0.0  # No citations possible

            # Calculate accuracy
            correct_citations = len(cited_pages & available_pages)
            accuracy = correct_citations / len(citations) if citations else 0.0

            return min(accuracy, 1.0)

        except Exception as e:
            logger.warning(f"Error calculating citation accuracy: {e}")
            return 0.5

    def _calculate_language_consistency(self, answer: str, chunk_contents: List[str]) -> float:
        """Calculate language consistency between answer and source chunks."""
        try:
            from langdetect import detect

            # Detect answer language
            try:
                answer_lang = detect(answer)
            except:
                answer_lang = 'unknown'

            # Detect chunk languages
            chunk_langs = []
            for chunk in chunk_contents[:3]:  # Check first 3 chunks
                try:
                    chunk_langs.append(detect(chunk))
                except:
                    chunk_langs.append('unknown')

            # Calculate consistency
            if chunk_langs and answer_lang != 'unknown':
                consistent_chunks = sum(1 for lang in chunk_langs if lang == answer_lang)
                return consistent_chunks / len(chunk_langs)
            else:
                return 0.5  # Neutral score

        except Exception as e:
            logger.warning(f"Error calculating language consistency: {e}")
            return 0.5

    def _calculate_retrieval_f1_score(self, query: str, chunks: List[Dict]) -> float:
        """Calculate F1 score for retrieval quality."""
        try:
            # Simplified F1 calculation based on query term coverage
            query_words = set(query.lower().split())
            retrieved_words = set()

            for chunk in chunks:
                content = chunk.get('page_content', '')
                retrieved_words.update(content.lower().split())

            # Calculate precision, recall, F1
            retrieved_relevant = len(query_words & retrieved_words)
            precision = retrieved_relevant / len(retrieved_words) if retrieved_words else 0.0
            recall = retrieved_relevant / len(query_words) if query_words else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            return min(f1, 1.0)

        except Exception as e:
            logger.warning(f"Error calculating retrieval F1: {e}")
            return 0.0

    def _calculate_response_completeness(self, query: str, answer: str, chunk_contents: List[str]) -> float:
        """Calculate how complete the response is."""
        try:
            # Combine all chunk content
            all_content = ' '.join(chunk_contents)

            # Check what percentage of available information was used
            content_words = set(all_content.lower().split())
            answer_words = set(answer.lower().split())

            used_words = len(content_words & answer_words)
            completeness = used_words / len(content_words) if content_words else 0.0

            # Boost for comprehensive answers (longer answers tend to be more complete)
            length_bonus = min(len(answer) / 500, 0.2)  # Max 0.2 bonus for 500+ char answers

            return min(completeness + length_bonus, 1.0)

        except Exception as e:
            logger.warning(f"Error calculating response completeness: {e}")
            return 0.0

    def _log_metrics(self, metrics: RAGMetrics):
        """Log metrics to file and console."""
        try:
            # Console logging
            logger.info(f"RAG Metrics - Query: '{metrics.query[:50]}...'")
            logger.info(f"  Response Time: {metrics.response_time:.2f}s")
            logger.info(f"  Chunks Retrieved/Used: {metrics.num_chunks_retrieved}/{metrics.num_chunks_used}")
            logger.info(f"  Context Relevance: {metrics.context_relevance_score:.2f}")
            logger.info(f"  Answer Relevance: {metrics.answer_relevance_score:.2f}")
            logger.info(f"  Citation Accuracy: {metrics.citation_accuracy:.2f}")
            logger.info(f"  Hallucination Score: {metrics.hallucination_score:.2f}")

            # File logging
            with open(self.log_file, 'a', encoding='utf-8') as f:
                json.dump(metrics.to_dict(), f, ensure_ascii=False)
                f.write('\n')

        except Exception as e:
            logger.error(f"Error logging metrics: {e}")

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics from all evaluations."""
        try:
            if not self.metrics_history:
                return {"total_queries": 0, "message": "No metrics available"}

            # Calculate averages
            total_queries = len(self.metrics_history)
            avg_response_time = np.mean([m.response_time for m in self.metrics_history])
            avg_context_relevance = np.mean([m.context_relevance_score for m in self.metrics_history])
            avg_answer_relevance = np.mean([m.answer_relevance_score for m in self.metrics_history])
            avg_citation_accuracy = np.mean([m.citation_accuracy for m in self.metrics_history])
            avg_hallucination = np.mean([m.hallucination_score for m in self.metrics_history])

            return {
                "total_queries": total_queries,
                "avg_response_time": round(avg_response_time, 2),
                "avg_context_relevance": round(avg_context_relevance, 2),
                "avg_answer_relevance": round(avg_answer_relevance, 2),
                "avg_citation_accuracy": round(avg_citation_accuracy, 2),
                "avg_hallucination_score": round(avg_hallucination, 2),
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error calculating summary stats: {e}")
            return {"error": str(e)}

    def export_metrics(self, filename: str):
        """Export all metrics to a JSON file."""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                metrics_data = [m.to_dict() for m in self.metrics_history]
                json.dump(metrics_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Exported {len(self.metrics_history)} metrics to {filename}")
        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")
