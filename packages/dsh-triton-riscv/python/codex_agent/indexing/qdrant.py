"""Legacy payload mapping; transport mechanics live in dsh-vector-retrieval."""
from dsh_vector_retrieval.qdrant import PayloadFields, Qdrant as VectorTransport, VectorServiceError


class Qdrant(VectorTransport):
    def __init__(self, definition, collection):
        super().__init__(definition, collection, fields=PayloadFields("memory_id", "revision"))

    def delete_revision(self, workspace, memory_id, revision, *, older=False, include=False):
        return self.delete_version(workspace, memory_id, revision, older=older, include=include)
