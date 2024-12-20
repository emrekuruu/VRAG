from datasets import load_dataset
from BasePipelines.voyage_pipeline import VoyagePipeline
import asyncio
from BasePipelines.config import Config
class FinanceBenchVoyagePipeline(VoyagePipeline):

    def prepare_dataset(self):
        data = load_dataset("PatronusAI/financebench")['train'].to_pandas()
        data['page_num'] = data['evidence'].apply(lambda x: x[0]['evidence_page_num'])
        return data

    async def process_query(self, data, idx):
        try:
            query = data.loc[idx, 'question']
            filename = data.loc[idx, 'doc_name'] + ".pdf"

            query_embedding = self.embedder.embed_query(query)

            results = await asyncio.to_thread(self.faiss_db.search, query_embedding, k=5, metadata_filter={"Filename": filename})

            qrels = {
                idx: {
                    result['id']: 1 / (1 + float(result['distance']))
                    for result in results
                }
            }

        except Exception as e:
            qrels = {idx: {}}

        return qrels

if __name__ == "__main__":
    async def main():
        config = Config(bucket_name="finance-bench")
        pipeline = FinanceBenchVoyagePipeline(config=config, task="FinanceBench")
        await pipeline()

    asyncio.run(main())
