from typing import Dict, Iterator, List, Tuple, AsyncIterator

import typer

from letta.agent_store.storage import StorageConnector
from letta.data_sources.connectors_helper import (
    assert_all_files_exist_locally,
    extract_metadata_from_files,
    get_filenames_in_dir,
)
from letta.embeddings import embedding_model
from letta.schemas.file import FileMetadata
from letta.schemas.passage import Passage
from letta.schemas.source import Source
from letta.utils import create_uuid_from_string


class AsyncDataConnector:
    """
    Base class for async data connectors that can be extended to generate files and passages from a custom data source.
    """

    async def find_files(self, source: Source) -> AsyncIterator[FileMetadata]:
        """
        Generate file metadata from a data source asynchronously.

        Returns:
            files (AsyncIterator[FileMetadata]): Generate file metadata for each file found.
        """
        raise NotImplementedError

    async def generate_passages(self, file: FileMetadata, chunk_size: int = 1024) -> AsyncIterator[Tuple[str, Dict]]:
        """
        Generate passage text and metadata from a list of files asynchronously.

        Args:
            file (FileMetadata): The document to generate passages from.
            chunk_size (int, optional): Chunk size for splitting passages. Defaults to 1024.

        Returns:
            passages (AsyncIterator[Tuple[str, Dict]]): Generate a tuple of string text and metadata dictionary for each passage.
        """
        raise NotImplementedError


async def load_data_async(
    connector: AsyncDataConnector,
    source: Source,
    passage_store: StorageConnector,
    file_metadata_store: StorageConnector,
):
    """Load data from a connector (generates file and passages) into a specified source_id, associated with a user_id."""
    embedding_config = source.embedding_config

    # embedding model
    embed_model = embedding_model(embedding_config)

    # insert passages/file
    passages = []
    embedding_to_document_name = {}
    passage_count = 0
    file_count = 0

    async for file_metadata in connector.find_files(source):
        file_count += 1
        await file_metadata_store.insert(file_metadata)

        # generate passages
        async for passage_text, passage_metadata in connector.generate_passages(file_metadata, chunk_size=embedding_config.embedding_chunk_size):
            # for some reason, llama index parsers sometimes return empty strings
            if len(passage_text) == 0:
                typer.secho(
                    f"Warning: Llama index parser returned empty string, skipping insert of passage with metadata '{passage_metadata}' into VectorDB. You can usually ignore this warning.",
                    fg=typer.colors.YELLOW,
                )
                continue

            # get embedding
            try:
                embedding = await embed_model.get_text_embedding_async(passage_text)
            except Exception as e:
                typer.secho(
                    f"Warning: Failed to get embedding for {passage_text} (error: {str(e)}), skipping insert into VectorDB.",
                    fg=typer.colors.YELLOW,
                )
                continue

            passage = Passage(
                id=create_uuid_from_string(f"{str(source.id)}_{passage_text}"),
                text=passage_text,
                file_id=file_metadata.id,
                source_id=source.id,
                metadata_=passage_metadata,
                user_id=source.user_id,
                embedding_config=source.embedding_config,
                embedding=embedding,
            )

            hashable_embedding = tuple(passage.embedding)
            file_name = file_metadata.file_name
            if hashable_embedding in embedding_to_document_name:
                typer.secho(
                    f"Warning: Duplicate embedding found for passage in {file_name} (already exists in {embedding_to_document_name[hashable_embedding]}), skipping insert into VectorDB.",
                    fg=typer.colors.YELLOW,
                )
                continue

            passages.append(passage)
            embedding_to_document_name[hashable_embedding] = file_name
            if len(passages) >= 100:
                # insert passages into passage store
                await passage_store.insert_many(passages)

                passage_count += len(passages)
                passages = []

    if len(passages) > 0:
        # insert passages into passage store
        await passage_store.insert_many(passages)
        passage_count += len(passages)

    return passage_count, file_count


class AsyncDirectoryConnector(AsyncDataConnector):
    """Async connector for loading files from a directory"""

    def __init__(self, input_dir: str = None, input_files: List[str] = None):
        """Initialize with either a directory path or list of file paths"""
        self.input_dir = input_dir
        self.input_files = input_files

        if input_dir is not None and input_files is not None:
            raise ValueError("Cannot specify both input_dir and input_files")
        if input_dir is None and input_files is None:
            raise ValueError("Must specify either input_dir or input_files")

        # get list of files
        if input_dir is not None:
            self.input_files = get_filenames_in_dir(input_dir)
        assert_all_files_exist_locally(self.input_files)

    async def find_files(self, source: Source) -> AsyncIterator[FileMetadata]:
        """Generate file metadata from files in directory"""
        for file_metadata in extract_metadata_from_files(self.input_files):
            yield file_metadata

    async def generate_passages(self, file: FileMetadata, chunk_size: int = 1024) -> AsyncIterator[Tuple[str, Dict]]:
        """Generate passage text and metadata from files"""
        from llama_index.core import SimpleDirectoryReader
        from llama_index.core.node_parser import TokenTextSplitter

        parser = TokenTextSplitter(chunk_size=chunk_size)
        documents = SimpleDirectoryReader(input_files=[file.file_path]).load_data()
        nodes = parser.get_nodes_from_documents(documents)
        for node in nodes:
            yield node.text, None


# Keep the original sync versions for backwards compatibility
class DataConnector:
    """
    Base class for data connectors that can be extended to generate files and passages from a custom data source.
    """

    def find_files(self, source: Source) -> Iterator[FileMetadata]:
        """
        Generate file metadata from a data source.

        Returns:
            files (Iterator[FileMetadata]): Generate file metadata for each file found.
        """
        raise NotImplementedError

    def generate_passages(self, file: FileMetadata, chunk_size: int = 1024) -> Iterator[Tuple[str, Dict]]:
        """
        Generate passage text and metadata from a list of files.

        Args:
            file (FileMetadata): The document to generate passages from.
            chunk_size (int, optional): Chunk size for splitting passages. Defaults to 1024.

        Returns:
            passages (Iterator[Tuple[str, Dict]]): Generate a tuple of string text and metadata dictionary for each passage.
        """
        raise NotImplementedError


def load_data(
    connector: DataConnector,
    source: Source,
    passage_store: StorageConnector,
    file_metadata_store: StorageConnector,
):
    """Load data from a connector (generates file and passages) into a specified source_id, associated with a user_id."""
    embedding_config = source.embedding_config

    # embedding model
    embed_model = embedding_model(embedding_config)

    # insert passages/file
    passages = []
    embedding_to_document_name = {}
    passage_count = 0
    file_count = 0
    for file_metadata in connector.find_files(source):
        file_count += 1
        file_metadata_store.insert(file_metadata)

        # generate passages
        for passage_text, passage_metadata in connector.generate_passages(file_metadata, chunk_size=embedding_config.embedding_chunk_size):
            # for some reason, llama index parsers sometimes return empty strings
            if len(passage_text) == 0:
                typer.secho(
                    f"Warning: Llama index parser returned empty string, skipping insert of passage with metadata '{passage_metadata}' into VectorDB. You can usually ignore this warning.",
                    fg=typer.colors.YELLOW,
                )
                continue

            # get embedding
            try:
                embedding = embed_model.get_text_embedding(passage_text)
            except Exception as e:
                typer.secho(
                    f"Warning: Failed to get embedding for {passage_text} (error: {str(e)}), skipping insert into VectorDB.",
                    fg=typer.colors.YELLOW,
                )
                continue

            passage = Passage(
                id=create_uuid_from_string(f"{str(source.id)}_{passage_text}"),
                text=passage_text,
                file_id=file_metadata.id,
                source_id=source.id,
                metadata_=passage_metadata,
                user_id=source.user_id,
                embedding_config=source.embedding_config,
                embedding=embedding,
            )

            hashable_embedding = tuple(passage.embedding)
            file_name = file_metadata.file_name
            if hashable_embedding in embedding_to_document_name:
                typer.secho(
                    f"Warning: Duplicate embedding found for passage in {file_name} (already exists in {embedding_to_document_name[hashable_embedding]}), skipping insert into VectorDB.",
                    fg=typer.colors.YELLOW,
                )
                continue

            passages.append(passage)
            embedding_to_document_name[hashable_embedding] = file_name
            if len(passages) >= 100:
                # insert passages into passage store
                passage_store.insert_many(passages)

                passage_count += len(passages)
                passages = []

    if len(passages) > 0:
        # insert passages into passage store
        passage_store.insert_many(passages)
        passage_count += len(passages)

    return passage_count, file_count


class DirectoryConnector(DataConnector):
    """Connector for loading files from a directory"""

    def __init__(self, input_dir: str = None, input_files: List[str] = None):
        """Initialize with either a directory path or list of file paths"""
        self.input_dir = input_dir
        self.input_files = input_files

        if input_dir is not None and input_files is not None:
            raise ValueError("Cannot specify both input_dir and input_files")
        if input_dir is None and input_files is None:
            raise ValueError("Must specify either input_dir or input_files")

        # get list of files
        if input_dir is not None:
            self.input_files = get_filenames_in_dir(input_dir)
        assert_all_files_exist_locally(self.input_files)

    def find_files(self, source: Source) -> Iterator[FileMetadata]:
        """Generate file metadata from files in directory"""
        for file_metadata in extract_metadata_from_files(self.input_files):
            yield file_metadata

    def generate_passages(self, file: FileMetadata, chunk_size: int = 1024) -> Iterator[Tuple[str, Dict]]:
        """Generate passage text and metadata from files"""
        from llama_index.core import SimpleDirectoryReader
        from llama_index.core.node_parser import TokenTextSplitter

        parser = TokenTextSplitter(chunk_size=chunk_size)
        documents = SimpleDirectoryReader(input_files=[file.file_path]).load_data()
        nodes = parser.get_nodes_from_documents(documents)
        for node in nodes:
            yield node.text, None
