import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { uploadToPresignedPost } from "../lib/storage";

// A minimal FormData stand-in that records append order (Node 22 has a global
// FormData, but we want to assert field order and the file-last rule).
class RecordingFormData {
  entries: [string, unknown][] = [];
  append(k: string, v: unknown) {
    this.entries.push([k, v]);
  }
}

// A fake XHR that records set headers and the sent body, then fires load.
class FakeXHR {
  static instances: FakeXHR[] = [];
  method = "";
  url = "";
  headers: Record<string, string> = {};
  sentBody: unknown = null;
  status = 204;
  responseText = "";
  statusText = "No Content";
  upload: { onprogress: ((e: unknown) => void) | null } = { onprogress: null };
  onload: (() => void) | null = null;
  onerror: (() => void) | null = null;
  onabort: (() => void) | null = null;

  constructor() {
    FakeXHR.instances.push(this);
  }
  open(method: string, url: string) {
    this.method = method;
    this.url = url;
  }
  setRequestHeader(k: string, v: string) {
    this.headers[k] = v;
  }
  send(body: unknown) {
    this.sentBody = body;
    queueMicrotask(() => this.onload && this.onload());
  }
  abort() {
    this.onabort && this.onabort();
  }
}

describe("storage client", () => {
  const realXHR = globalThis.XMLHttpRequest;
  const realFD = globalThis.FormData;

  beforeEach(() => {
    FakeXHR.instances = [];
    // @ts-expect-error test stub
    globalThis.XMLHttpRequest = FakeXHR;
    // @ts-expect-error test stub
    globalThis.FormData = RecordingFormData;
  });
  afterEach(() => {
    globalThis.XMLHttpRequest = realXHR;
    globalThis.FormData = realFD;
  });

  it("sends exactly the presigned fields, file last, and NO headers", async () => {
    const post = {
      url: "https://s3.example.test/bucket",
      key: "scans/x/input/video.mov",
      fields: {
        key: "scans/x/input/video.mov",
        "Content-Type": "video/quicktime",
        policy: "base64policy",
        "x-amz-signature": "sig",
      },
    };
    const file = { name: "video.mov" } as unknown as File;

    await uploadToPresignedPost(post, file);

    expect(FakeXHR.instances.length).toBe(1);
    const xhr = FakeXHR.instances[0];
    expect(xhr.method).toBe("POST");
    expect(xhr.url).toBe(post.url);
    // No request headers were set by the client (S3 authenticates from fields).
    expect(Object.keys(xhr.headers)).toEqual([]);

    const fd = xhr.sentBody as RecordingFormData;
    const keys = fd.entries.map((e) => e[0]);
    // All fields present, in order, then "file" LAST.
    expect(keys).toEqual([
      "key",
      "Content-Type",
      "policy",
      "x-amz-signature",
      "file",
    ]);
    expect(fd.entries[fd.entries.length - 1][0]).toBe("file");
    expect(fd.entries[fd.entries.length - 1][1]).toBe(file);
  });

  it("rejects on a non-2xx S3 response", async () => {
    const post = { url: "https://s3.example.test/b", key: "k", fields: {} };
    const p = uploadToPresignedPost(post, {} as File);
    const xhr = FakeXHR.instances[0];
    xhr.status = 403;
    xhr.responseText = "AccessDenied";
    // trigger load with the error status
    xhr.onload && xhr.onload();
    await expect(p).rejects.toThrow(/403/);
  });
});

describe("assetUrl with signed URLs", () => {
  it("keeps API-signed URLs unchanged and prefixes bare keys", async () => {
    const { assetUrl } = await import("../lib/storage");
    expect(assetUrl("/api/assets/scans/s/mesh/object.glb?exp=1&sig=a")).toBe("/api/assets/scans/s/mesh/object.glb?exp=1&sig=a");
    expect(assetUrl("scans/s/mesh/object.glb")).toBe("/api/assets/scans/s/mesh/object.glb");
    expect(assetUrl("https://bucket.s3.amazonaws.com/x?X-Amz-Signature=1")).toBe("https://bucket.s3.amazonaws.com/x?X-Amz-Signature=1");
  });
});
