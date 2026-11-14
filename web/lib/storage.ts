// The storage client. SEPARATE from the API client on purpose (SPEC 3.3 /
// AGENTS.md decision 1): it sends EXACTLY the pre-signed POST form fields to the
// given url, with the file appended LAST, and adds NO extra headers (no
// Authorization, no Content-Type -- the browser sets the multipart boundary
// itself). S3 authenticates the request from the form fields; a stray header
// would make it reject the upload. Asset GETs likewise use the signed URL as
// given.

import type { CreateScanResponse } from "./contracts.gen";

type PresignedPost = CreateScanResponse["uploads"][number];

export interface UploadProgress {
  loaded: number;
  total: number;
}

// Upload one file to a pre-signed POST target. Fields are appended in the exact
// order given, then the file as the "file" field LAST (S3 requires the file to
// be the last multipart part). Uses XHR so we can report progress; fetch has no
// upload-progress event. No headers are set explicitly.
export function uploadToPresignedPost(
  post: PresignedPost,
  file: File | Blob,
  onProgress?: (p: UploadProgress) => void,
  signal?: AbortSignal,
): Promise<void> {
  return new Promise((resolve, reject) => {
    const form = new FormData();
    for (const [key, value] of Object.entries(post.fields)) {
      form.append(key, value);
    }
    // File MUST be appended last.
    form.append("file", file);

    const xhr = new XMLHttpRequest();
    xhr.open("POST", post.url, true);
    // Deliberately set NO request headers: FormData drives Content-Type and S3
    // authenticates from the form fields alone.

    if (onProgress && xhr.upload) {
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) {
          onProgress({ loaded: e.loaded, total: e.total });
        }
      };
    }

    xhr.onload = () => {
      // S3 returns 201/204 on a successful pre-signed POST.
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve();
      } else {
        reject(
          new Error(
            `Upload failed (${xhr.status}): ${xhr.responseText || xhr.statusText}`,
          ),
        );
      }
    };
    xhr.onerror = () => reject(new Error("Upload network error"));
    xhr.onabort = () => reject(new DOMException("Upload aborted", "AbortError"));

    if (signal) {
      if (signal.aborted) {
        xhr.abort();
        return;
      }
      signal.addEventListener("abort", () => xhr.abort(), { once: true });
    }

    xhr.send(form);
  });
}

// Fetch a signed asset URL as given, with no added headers. Returns the Blob.
export async function fetchAsset(signedUrl: string): Promise<Blob> {
  const res = await fetch(signedUrl);
  if (!res.ok) {
    throw new Error(`Asset fetch failed (${res.status}) for ${signedUrl}`);
  }
  return res.blob();
}

// Resolve a storage key (e.g. "scans/scn/mesh/object.glb") to a URL the browser
// can load. In dev the API serves assets under /api/assets/<key>; a fully signed
// URL (starting with http) is returned unchanged.
export function assetUrl(uriOrKey: string): string {
  if (/^https?:\/\//i.test(uriOrKey)) return uriOrKey;
  // The API returns signed URLs (SPEC 5), already rooted at /api/...; only bare storage keys get a prefix.
  if (uriOrKey.startsWith("/api/")) return uriOrKey;
  const clean = uriOrKey.replace(/^\/+/, "");
  return `/api/assets/${clean}`;
}
