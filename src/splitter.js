import "dotenv/config";
import "cheerio";
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";

const model = new ChatOpenAI({
  temperature: 0,
  model: process.env.OPENAI_BASE_MODEL,
  apiKey: process.env.OPENAI_API_KEY,
  baseURL: process.env.OPENAI_BASE_URL,
});

const embeddings = new OpenAIEmbeddings({
  model: process.env.OPENAI_EMBEDDINGS_MODEL,
  apiKey: process.env.OPENAI_API_KEY,
  baseURL: process.env.OPENAI_BASE_URL,
});

const cheerioLoader = new CheerioWebBaseLoader(
  "https://juejin.cn/post/7587684397530595355",
  { selector: ".main-area p" },
);

const documents = await cheerioLoader.load();

const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 200,
  chunkOverlap: 20,
  separators: ["\n", "。", "，"],
});

const splitDocuments = await textSplitter.splitDocuments(documents);

console.log("splitDocuments", splitDocuments);

console.log("文档分割完成", `共${splitDocuments.length}个文档`);

const vectorStore = await MemoryVectorStore.fromDocuments(
  splitDocuments,
  embeddings,
);

const retriever = vectorStore.asRetriever({ k: 2 });

const questions = ["这篇文章讲了什么？"];

for (const question of questions) {
  console.log("=".repeat(80));
  console.log(`问题：${question}`);
  console.log("=".repeat(80));

  const retrievedDocs = await retriever.invoke(question);

  const scoredResults = await vectorStore.similaritySearchWithScore(
    question,
    3,
  );

  console.log("\n检索到的文档及相似度评分");

  retrievedDocs.forEach((doc, i) => {
    // 找到对应的评分
    const scoredResult = scoredResults.find(
      ([scoredDoc]) => scoredDoc.pageContent === doc.pageContent,
    );
    const score = scoredResult ? scoredResult[1] : null;
    const similarity = score !== null ? (1 - score).toFixed(4) : "N/A";

    console.log(`\n[文档 ${i + 1}] 相似度: ${similarity}`);
    console.log(`内容: ${doc.pageContent}`);
    console.log(
      `元数据: 章节=${doc.metadata.chapter}, 角色=${doc.metadata.character}, 类型=${doc.metadata.type}, 心情=${doc.metadata.mood}`,
    );
  });
  // 构建 prompt
  const context = retrievedDocs
    .map((doc, i) => `[片段${i + 1}]\n${doc.pageContent}`)
    .join("\n\n━━━━━\n\n");

  const prompt = `你是一个文档辅助阅读助手，根据文章内容来解答:
${context}

问题: ${question}

你的回答:`;

  // 直接使用 model.invoke
  console.log("\n【AI 回答】");
  const response = await model.invoke(prompt);
  console.log(response.content);
  console.log("\n");
}
