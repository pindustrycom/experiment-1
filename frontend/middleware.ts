import { NextResponse, type NextRequest } from "next/server";

// Cheap gate: pages require the session cookie to exist; the API validates it on every call and the
// client redirects to /login on 401.
export function middleware(req: NextRequest) {
  const { pathname } = req.nextUrl;
  if (pathname.startsWith("/login") || pathname.startsWith("/api") || pathname.startsWith("/_next") || pathname === "/favicon.svg") {
    return NextResponse.next();
  }
  if (!req.cookies.get("sm_session")) {
    const url = req.nextUrl.clone();
    url.pathname = "/login";
    url.searchParams.set("next", pathname);
    return NextResponse.redirect(url);
  }
  return NextResponse.next();
}

export const config = { matcher: ["/((?!_next/static|_next/image).*)"] };
