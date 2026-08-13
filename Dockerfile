# Debian rather than Alpine: onnxruntime-node ships glibc binaries and will not
# load against musl. Node 24 runs the TypeScript sources directly, so there is
# no build stage and nothing to keep in sync between source and artefact.
FROM node:24-slim

WORKDIR /app
ENV NODE_ENV=production \
    HOST=0.0.0.0 \
    PORT=8080

# Dependencies first, so a source edit does not re-resolve the tree.
COPY package.json package-lock.json* ./
RUN npm ci --omit=dev --no-audit --no-fund

# The encoder and reranker are vendored into the image at build time rather
# than downloaded at boot — a container that reaches out to Hugging Face on its
# first request is a container whose first request is slow and whose startup
# depends on someone else's uptime.
COPY src ./src
RUN node src/tools/fetch-model.ts

COPY public ./public
COPY data/index ./data/index
COPY data/raw/queries.jsonl ./data/raw/queries.jsonl

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=4s --start-period=40s --retries=3 \
  CMD node -e "fetch('http://127.0.0.1:'+(process.env.PORT||8080)+'/api/health').then(r=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))"

# The index and both ONNX graphs load and warm before the socket opens, so the
# first request a visitor makes is already a warm one.
CMD ["node", "src/server/index.ts"]
