module.exports = [
"[externals]/next/dist/compiled/next-server/app-route-turbo.runtime.dev.js [external] (next/dist/compiled/next-server/app-route-turbo.runtime.dev.js, cjs)", ((__turbopack_context__, module, exports) => {

const mod = __turbopack_context__.x("next/dist/compiled/next-server/app-route-turbo.runtime.dev.js", () => require("next/dist/compiled/next-server/app-route-turbo.runtime.dev.js"));

module.exports = mod;
}),
"[externals]/next/dist/compiled/@opentelemetry/api [external] (next/dist/compiled/@opentelemetry/api, cjs)", ((__turbopack_context__, module, exports) => {

const mod = __turbopack_context__.x("next/dist/compiled/@opentelemetry/api", () => require("next/dist/compiled/@opentelemetry/api"));

module.exports = mod;
}),
"[externals]/next/dist/compiled/next-server/app-page-turbo.runtime.dev.js [external] (next/dist/compiled/next-server/app-page-turbo.runtime.dev.js, cjs)", ((__turbopack_context__, module, exports) => {

const mod = __turbopack_context__.x("next/dist/compiled/next-server/app-page-turbo.runtime.dev.js", () => require("next/dist/compiled/next-server/app-page-turbo.runtime.dev.js"));

module.exports = mod;
}),
"[externals]/next/dist/server/app-render/work-unit-async-storage.external.js [external] (next/dist/server/app-render/work-unit-async-storage.external.js, cjs)", ((__turbopack_context__, module, exports) => {

const mod = __turbopack_context__.x("next/dist/server/app-render/work-unit-async-storage.external.js", () => require("next/dist/server/app-render/work-unit-async-storage.external.js"));

module.exports = mod;
}),
"[externals]/next/dist/server/app-render/work-async-storage.external.js [external] (next/dist/server/app-render/work-async-storage.external.js, cjs)", ((__turbopack_context__, module, exports) => {

const mod = __turbopack_context__.x("next/dist/server/app-render/work-async-storage.external.js", () => require("next/dist/server/app-render/work-async-storage.external.js"));

module.exports = mod;
}),
"[externals]/next/dist/shared/lib/no-fallback-error.external.js [external] (next/dist/shared/lib/no-fallback-error.external.js, cjs)", ((__turbopack_context__, module, exports) => {

const mod = __turbopack_context__.x("next/dist/shared/lib/no-fallback-error.external.js", () => require("next/dist/shared/lib/no-fallback-error.external.js"));

module.exports = mod;
}),
"[externals]/next/dist/server/app-render/after-task-async-storage.external.js [external] (next/dist/server/app-render/after-task-async-storage.external.js, cjs)", ((__turbopack_context__, module, exports) => {

const mod = __turbopack_context__.x("next/dist/server/app-render/after-task-async-storage.external.js", () => require("next/dist/server/app-render/after-task-async-storage.external.js"));

module.exports = mod;
}),
"[project]/codes/python/axtone/frontend/axtone/app/api/convert/route.ts [app-route] (ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "POST",
    ()=>POST
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$server$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/codes/python/axtone/frontend/axtone/node_modules/next/server.js [app-route] (ecmascript)");
;
async function POST(request) {
    try {
        const formData = await request.formData();
        // Get API URL from environment variable
        const PYTHON_API_URL = process.env.PYTHON_API_URL || ("TURBOPACK compile-time value", "https://axtone.onrender.com");
        if ("TURBOPACK compile-time falsy", 0) //TURBOPACK unreachable
        ;
        console.log('Connecting to:', PYTHON_API_URL);
        // Forward the request to the Python FastAPI backend with extended timeout
        const controller = new AbortController();
        const timeoutId = setTimeout(()=>controller.abort(), 180000); // 3 minute timeout for server wake-up
        try {
            const response = await fetch(`${PYTHON_API_URL}/api/convert`, {
                method: 'POST',
                body: formData,
                signal: controller.signal
            });
            clearTimeout(timeoutId);
            console.log('✅ Response received:', response.status);
            if (!response.ok) {
                const error = await response.json();
                console.error('Backend error:', error);
                return __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$server$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["NextResponse"].json({
                    error: error.detail || 'Conversion failed'
                }, {
                    status: response.status
                });
            }
            const data = await response.json();
            console.log('✅ Conversion successful');
            return __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$server$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["NextResponse"].json(data);
        } catch (fetchError) {
            clearTimeout(timeoutId);
            if (fetchError instanceof Error && fetchError.name === 'AbortError') {
                throw new Error('Request timed out. Please try again.');
            }
            throw new Error(`Cannot connect to server: ${fetchError instanceof Error ? fetchError.message : 'Network error'}`);
        }
    } catch (error) {
        console.error('API route error:', error);
        return __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$server$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["NextResponse"].json({
            error: error instanceof Error ? error.message : 'Failed to connect to processing server'
        }, {
            status: 500
        });
    }
}
}),
];

//# sourceMappingURL=%5Broot-of-the-server%5D__3ac00206._.js.map