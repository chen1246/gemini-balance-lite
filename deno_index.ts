import { handleRequest } from "./handle_request.js";

async function denoHandleRequest(req: Request): Promise<Response> {
  const url = new URL(req.url);
  console.log('Request URL:', req.url);
  return handleRequest(req);
};

// 注意:新 Deno Deploy 平台不允许显式指定端口,必须用无参 Deno.serve(handler)
Deno.serve(denoHandleRequest);