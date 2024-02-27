"""
This module provides functions for generating a highlighted PDF with important sentences.

The main function, `generate_highlighted_pdf`, takes an input PDF file and a pre-trained
sentence embedding model as input.

It splits the text of the PDF into sentences, computes sentence embeddings, and builds a
graph based on the cosine similarity between embeddings and at the same time split the
sentences to different clusters using clustering.

The sentences are then ranked using PageRank scores and a the middle of the cluster,
and important sentences are selected based on a threshold and clustering.

Finally, the selected sentences are highlighted in the PDF and the highlighted PDF content
is returned.

Other utility functions in this module include functions for loading a sentence embedding
model, encoding sentences, computing similarity matrices,building graphs, ranking sentences,
clustering sentence embeddings, and splitting text into sentences.

Note: This module requires the PyMuPDF, networkx, numpy, torch, sentence_transformers, and
sklearn libraries to be installed.
"""

import logging
from typing import BinaryIO, List, Tuple

import fitz  # PyMuPDF
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

# Constants
MAX_PAGE = 40
MAX_SENTENCES = 2000
PAGERANK_THRESHOLD_RATIO = 0.15
NUM_CLUSTERS_RATIO = 0.05
MIN_WORDS = 10

# Logger configuration
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


def load_sentence_model(revision: str = None) -> SentenceTransformer:
    """
    Load a pre-trained sentence embedding model.

    Args:
        revision (str): Optional parameter to specify the model revision.

    Returns:
        SentenceTransformer: A pre-trained sentence embedding model.
    """
    return SentenceTransformer("avsolatorio/GIST-Embedding-v0", revision=revision)


def encode_sentence(model: SentenceTransformer, sentence: str) -> torch.Tensor:
    """
    Encode a sentence into a fixed-dimensional vector representation.

    Args:
        model (SentenceTransformer): A pre-trained sentence embedding model.
        sentence (str): Input sentence.

    Returns:
        torch.Tensor: Encoded sentence vector.
    """

    model.eval()  # Set the model to evaluation mode

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():  # Disable gradient tracking
        return model.encode(sentence, convert_to_tensor=True).to(device)


def compute_similarity_matrix(embeddings: torch.Tensor) -> np.ndarray:
    """
    Compute the cosine similarity matrix between sentence embeddings.

    Args:
        embeddings (torch.Tensor): Sentence embeddings.

    Returns:
        np.ndarray: Cosine similarity matrix.
    """
    scores = F.cosine_similarity(
        embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1
    )
    similarity_matrix = scores.cpu().numpy()
    normalized_adjacency_matrix = similarity_matrix / similarity_matrix.sum(
        axis=1, keepdims=True
    )
    return normalized_adjacency_matrix


def build_graph(normalized_adjacency_matrix: np.ndarray) -> nx.DiGraph:
    """
    Build a directed graph from a normalized adjacency matrix.

    Args:
        normalized_adjacency_matrix (np.ndarray): Normalized adjacency matrix.

    Returns:
        nx.DiGraph: Directed graph.
    """
    return nx.DiGraph(normalized_adjacency_matrix)


def rank_sentences(graph: nx.DiGraph, sentences: List[str]) -> List[Tuple[str, float]]:
    """
    Rank sentences based on PageRank scores.

    Args:
        graph (nx.DiGraph): Directed graph.
        sentences (List[str]): List of sentences.

    Returns:
        List[Tuple[str, float]]: Ranked sentences with their PageRank scores.
    """
    pagerank_scores = nx.pagerank(graph)
    ranked_sentences = sorted(
        zip(sentences, pagerank_scores.values()),
        key=lambda x: x[1],
        reverse=True,
    )
    return ranked_sentences


def cluster_sentences(
    embeddings: torch.Tensor, num_clusters: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cluster sentence embeddings using K-means clustering.

    Args:
        embeddings (torch.Tensor): Sentence embeddings.
        num_clusters (int): Number of clusters.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Cluster assignments and cluster centers.
    """
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_assignments = kmeans.fit_predict(embeddings.cpu())
    cluster_centers = kmeans.cluster_centers_
    return cluster_assignments, cluster_centers


def get_middle_sentence(cluster_indices: np.ndarray, sentences: List[str]) -> List[str]:
    """
    Get the middle sentence from each cluster.

    Args:
        cluster_indices (np.ndarray): Cluster assignments.
        sentences (List[str]): List of sentences.

    Returns:
        List[str]: Middle sentences from each cluster.
    """
    middle_indices = [
        int(np.median(np.where(cluster_indices == i)[0]))
        for i in range(max(cluster_indices) + 1)
    ]
    middle_sentences = [sentences[i] for i in middle_indices]
    return middle_sentences


def split_text_into_sentences(text: str, min_words: int = MIN_WORDS) -> List[str]:
    """
    Split text into sentences.

    Args:
        text (str): Input text.
        min_words (int): Minimum number of words for a valid sentence.

    Returns:
        List[str]: List of sentences.
    """
    sentences = [
        s.strip() for s in text.split(".") if s.strip() and len(s.split()) >= min_words
    ]
    return sentences


def extract_text_from_pages(doc):
    """Generator to yield text per page from the PDF, for memory efficiency for large PDFs."""
    for page_num in range(len(doc)):
        yield doc[page_num].get_text()


def generate_highlighted_pdf(
    input_pdf_file: BinaryIO, model=load_sentence_model()
) -> bytes:
    """
    Generate a highlighted PDF with important sentences.

    Args:
        input_pdf_file: Input PDF file object.
        model (SentenceTransformer): Pre-trained sentence embedding model.

    Returns:
        bytes: Highlighted PDF content.
    """
    with fitz.open(stream=input_pdf_file.read(), filetype="pdf") as doc:
        num_pages = doc.page_count

        if num_pages > MAX_PAGE:
            # It will show the error message for the user.
            return f"The PDF file exceeds the maximum limit of {MAX_PAGE} pages."

        sentences = []
        for page_text in extract_text_from_pages(doc):  # Memory efficient
            sentences.extend(split_text_into_sentences(page_text))

        len_sentences = len(sentences)

        print(len_sentences)

        if len_sentences > MAX_SENTENCES:
            # It will show the error message for the user.
            return (
                f"The PDF file exceeds the maximum limit of {MAX_SENTENCES} sentences."
            )

        embeddings = encode_sentence(model, sentences)
        similarity_matrix = compute_similarity_matrix(embeddings)
        graph = build_graph(similarity_matrix)
        ranked_sentences = rank_sentences(graph, sentences)

        pagerank_threshold = int(len(ranked_sentences) * PAGERANK_THRESHOLD_RATIO) + 1
        top_pagerank_sentences = [
            sentence[0] for sentence in ranked_sentences[:pagerank_threshold]
        ]

        num_clusters = int(len_sentences * NUM_CLUSTERS_RATIO) + 1
        cluster_assignments, _ = cluster_sentences(embeddings, num_clusters)

        center_sentences = get_middle_sentence(cluster_assignments, sentences)
        important_sentences = list(set(top_pagerank_sentences + center_sentences))

        for i in range(num_pages):
            try:
                page = doc[i]

                for sentence in important_sentences:
                    rects = page.search_for(sentence)
                    colors = (fitz.pdfcolor["yellow"], fitz.pdfcolor["green"])

                    for i, rect in enumerate(rects):
                        color = colors[i % 2]
                        annot = page.add_highlight_annot(rect)
                        annot.set_colors(stroke=color)
                        annot.update()
            except Exception as e:
                logger.error(f"Error processing page {i}: {e}")

        output_pdf = doc.write()

    return output_pdf
