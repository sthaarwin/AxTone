module.exports = [
"[project]/codes/python/axtone/frontend/axtone/lib/utils.ts [app-ssr] (ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "cn",
    ()=>cn
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$clsx$2f$dist$2f$clsx$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/codes/python/axtone/frontend/axtone/node_modules/clsx/dist/clsx.mjs [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$tailwind$2d$merge$2f$dist$2f$bundle$2d$mjs$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/codes/python/axtone/frontend/axtone/node_modules/tailwind-merge/dist/bundle-mjs.mjs [app-ssr] (ecmascript)");
;
;
function cn(...inputs) {
    return (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$tailwind$2d$merge$2f$dist$2f$bundle$2d$mjs$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["twMerge"])((0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$clsx$2f$dist$2f$clsx$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["clsx"])(inputs));
}
}),
"[project]/codes/python/axtone/frontend/axtone/components/ui/button.tsx [app-ssr] (ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "Button",
    ()=>Button,
    "buttonVariants",
    ()=>buttonVariants
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/codes/python/axtone/frontend/axtone/node_modules/next/dist/server/route-modules/app-page/vendored/ssr/react-jsx-dev-runtime.js [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f40$radix$2d$ui$2f$react$2d$slot$2f$dist$2f$index$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/codes/python/axtone/frontend/axtone/node_modules/@radix-ui/react-slot/dist/index.mjs [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$class$2d$variance$2d$authority$2f$dist$2f$index$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/codes/python/axtone/frontend/axtone/node_modules/class-variance-authority/dist/index.mjs [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$lib$2f$utils$2e$ts__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/codes/python/axtone/frontend/axtone/lib/utils.ts [app-ssr] (ecmascript)");
;
;
;
;
const buttonVariants = (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$class$2d$variance$2d$authority$2f$dist$2f$index$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["cva"])("inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-md text-sm font-medium transition-all disabled:pointer-events-none disabled:opacity-50 [&_svg]:pointer-events-none [&_svg:not([class*='size-'])]:size-4 shrink-0 [&_svg]:shrink-0 outline-none focus-visible:border-ring focus-visible:ring-ring/50 focus-visible:ring-[3px] aria-invalid:ring-destructive/20 dark:aria-invalid:ring-destructive/40 aria-invalid:border-destructive", {
    variants: {
        variant: {
            default: "bg-primary text-primary-foreground hover:bg-primary/90",
            destructive: "bg-destructive text-white hover:bg-destructive/90 focus-visible:ring-destructive/20 dark:focus-visible:ring-destructive/40 dark:bg-destructive/60",
            outline: "border bg-background shadow-xs hover:bg-accent hover:text-accent-foreground dark:bg-input/30 dark:border-input dark:hover:bg-input/50",
            secondary: "bg-secondary text-secondary-foreground hover:bg-secondary/80",
            ghost: "hover:bg-accent hover:text-accent-foreground dark:hover:bg-accent/50",
            link: "text-primary underline-offset-4 hover:underline"
        },
        size: {
            default: "h-9 px-4 py-2 has-[>svg]:px-3",
            xs: "h-6 gap-1 rounded-md px-2 text-xs has-[>svg]:px-1.5 [&_svg:not([class*='size-'])]:size-3",
            sm: "h-8 rounded-md gap-1.5 px-3 has-[>svg]:px-2.5",
            lg: "h-10 rounded-md px-6 has-[>svg]:px-4",
            icon: "size-9",
            "icon-xs": "size-6 rounded-md [&_svg:not([class*='size-'])]:size-3",
            "icon-sm": "size-8",
            "icon-lg": "size-10"
        }
    },
    defaultVariants: {
        variant: "default",
        size: "default"
    }
});
function Button({ className, variant = "default", size = "default", asChild = false, ...props }) {
    const Comp = asChild ? __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f40$radix$2d$ui$2f$react$2d$slot$2f$dist$2f$index$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["Slot"] : "button";
    return /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(Comp, {
        "data-slot": "button",
        "data-variant": variant,
        "data-size": size,
        className: (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$lib$2f$utils$2e$ts__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["cn"])(buttonVariants({
            variant,
            size,
            className
        })),
        ...props
    }, void 0, false, {
        fileName: "[project]/codes/python/axtone/frontend/axtone/components/ui/button.tsx",
        lineNumber: 54,
        columnNumber: 5
    }, this);
}
;
}),
"[project]/codes/python/axtone/frontend/axtone/components/file-uploader.tsx [app-ssr] (ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "default",
    ()=>FileUploader
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/codes/python/axtone/frontend/axtone/node_modules/next/dist/server/route-modules/app-page/vendored/ssr/react-jsx-dev-runtime.js [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/codes/python/axtone/frontend/axtone/node_modules/next/dist/server/route-modules/app-page/vendored/ssr/react.js [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$upload$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__$3c$export__default__as__Upload$3e$__ = __turbopack_context__.i("[project]/codes/python/axtone/frontend/axtone/node_modules/lucide-react/dist/esm/icons/upload.js [app-ssr] (ecmascript) <export default as Upload>");
var __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$music$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__$3c$export__default__as__Music$3e$__ = __turbopack_context__.i("[project]/codes/python/axtone/frontend/axtone/node_modules/lucide-react/dist/esm/icons/music.js [app-ssr] (ecmascript) <export default as Music>");
var __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$components$2f$ui$2f$button$2e$tsx__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/codes/python/axtone/frontend/axtone/components/ui/button.tsx [app-ssr] (ecmascript)");
'use client';
;
;
;
;
function FileUploader({ onFileSelect }) {
    const [isDragOver, setIsDragOver] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useState"])(false);
    const inputRef = (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useRef"])(null);
    const handleDragOver = (e)=>{
        e.preventDefault();
        setIsDragOver(true);
    };
    const handleDragLeave = ()=>{
        setIsDragOver(false);
    };
    const handleDrop = (e)=>{
        e.preventDefault();
        setIsDragOver(false);
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            const file = files[0];
            if (file.type.startsWith('audio/')) {
                onFileSelect(file);
            }
        }
    };
    const handleFileInput = (e)=>{
        const files = e.currentTarget.files;
        if (files && files.length > 0) {
            onFileSelect(files[0]);
        }
    };
    return /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
        onDragOver: handleDragOver,
        onDragLeave: handleDragLeave,
        onDrop: handleDrop,
        className: `relative rounded-2xl border-2 border-dashed transition-all duration-200 p-12 text-center cursor-pointer
        ${isDragOver ? 'border-cyan-400 bg-cyan-400/5' : 'border-slate-700 bg-slate-900/50 hover:border-slate-600'}`,
        onClick: ()=>inputRef.current?.click(),
        children: [
            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("input", {
                ref: inputRef,
                type: "file",
                accept: "audio/mp3,audio/wav,audio/mpeg,.mp3,.wav",
                onChange: handleFileInput,
                className: "hidden"
            }, void 0, false, {
                fileName: "[project]/codes/python/axtone/frontend/axtone/components/file-uploader.tsx",
                lineNumber: 58,
                columnNumber: 7
            }, this),
            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                className: "space-y-4",
                children: [
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                        className: "flex justify-center",
                        children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                            className: "w-16 h-16 bg-gradient-to-br from-cyan-400/20 to-blue-600/20 rounded-full flex items-center justify-center border border-cyan-400/30",
                            children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$upload$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__$3c$export__default__as__Upload$3e$__["Upload"], {
                                className: "w-8 h-8 text-cyan-400"
                            }, void 0, false, {
                                fileName: "[project]/codes/python/axtone/frontend/axtone/components/file-uploader.tsx",
                                lineNumber: 69,
                                columnNumber: 13
                            }, this)
                        }, void 0, false, {
                            fileName: "[project]/codes/python/axtone/frontend/axtone/components/file-uploader.tsx",
                            lineNumber: 68,
                            columnNumber: 11
                        }, this)
                    }, void 0, false, {
                        fileName: "[project]/codes/python/axtone/frontend/axtone/components/file-uploader.tsx",
                        lineNumber: 67,
                        columnNumber: 9
                    }, this),
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                        className: "space-y-2",
                        children: [
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("h3", {
                                className: "text-xl font-semibold text-white",
                                children: "Drop your audio file here"
                            }, void 0, false, {
                                fileName: "[project]/codes/python/axtone/frontend/axtone/components/file-uploader.tsx",
                                lineNumber: 74,
                                columnNumber: 11
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                                className: "text-slate-400",
                                children: "or click to select a file"
                            }, void 0, false, {
                                fileName: "[project]/codes/python/axtone/frontend/axtone/components/file-uploader.tsx",
                                lineNumber: 77,
                                columnNumber: 11
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                                className: "text-sm text-slate-500",
                                children: "Supports MP3 and WAV formats (Max 50MB)"
                            }, void 0, false, {
                                fileName: "[project]/codes/python/axtone/frontend/axtone/components/file-uploader.tsx",
                                lineNumber: 80,
                                columnNumber: 11
                            }, this)
                        ]
                    }, void 0, true, {
                        fileName: "[project]/codes/python/axtone/frontend/axtone/components/file-uploader.tsx",
                        lineNumber: 73,
                        columnNumber: 9
                    }, this),
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$components$2f$ui$2f$button$2e$tsx__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["Button"], {
                        className: "mt-6 bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-600 hover:to-blue-700 text-white border-0",
                        size: "lg",
                        children: [
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$music$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__$3c$export__default__as__Music$3e$__["Music"], {
                                className: "w-4 h-4 mr-2"
                            }, void 0, false, {
                                fileName: "[project]/codes/python/axtone/frontend/axtone/components/file-uploader.tsx",
                                lineNumber: 89,
                                columnNumber: 11
                            }, this),
                            "Choose File"
                        ]
                    }, void 0, true, {
                        fileName: "[project]/codes/python/axtone/frontend/axtone/components/file-uploader.tsx",
                        lineNumber: 85,
                        columnNumber: 9
                    }, this)
                ]
            }, void 0, true, {
                fileName: "[project]/codes/python/axtone/frontend/axtone/components/file-uploader.tsx",
                lineNumber: 66,
                columnNumber: 7
            }, this)
        ]
    }, void 0, true, {
        fileName: "[project]/codes/python/axtone/frontend/axtone/components/file-uploader.tsx",
        lineNumber: 47,
        columnNumber: 5
    }, this);
}
}),
"[project]/codes/python/axtone/frontend/axtone/components/processing-state.tsx [app-ssr] (ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "default",
    ()=>ProcessingState
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/codes/python/axtone/frontend/axtone/node_modules/next/dist/server/route-modules/app-page/vendored/ssr/react-jsx-dev-runtime.js [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/codes/python/axtone/frontend/axtone/node_modules/next/dist/server/route-modules/app-page/vendored/ssr/react.js [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$loader$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__$3c$export__default__as__Loader$3e$__ = __turbopack_context__.i("[project]/codes/python/axtone/frontend/axtone/node_modules/lucide-react/dist/esm/icons/loader.js [app-ssr] (ecmascript) <export default as Loader>");
'use client';
;
;
;
function ProcessingState({ fileName }) {
    const [steps, setSteps] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useState"])([
        {
            id: 'extract',
            label: 'Extracting Frequencies',
            completed: false
        },
        {
            id: 'graph',
            label: 'Building Fretboard Graph',
            completed: false
        },
        {
            id: 'dijkstra',
            label: 'Running Dijkstra Optimization',
            completed: false
        }
    ]);
    (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useEffect"])(()=>{
        // Animate steps completion
        const timings = [
            800,
            2000,
            3200
        ];
        timings.forEach((timing, index)=>{
            setTimeout(()=>{
                setSteps((prev)=>prev.map((step, i)=>i === index ? {
                            ...step,
                            completed: true
                        } : step));
            }, timing);
        });
    }, []);
    return /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
        className: "flex flex-col items-center justify-center min-h-[600px] space-y-12",
        children: [
            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                className: "relative w-24 h-24",
                children: [
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                        className: "absolute inset-0 bg-gradient-to-br from-cyan-400 to-blue-600 rounded-lg blur-xl opacity-30 animate-pulse"
                    }, void 0, false, {
                        fileName: "[project]/codes/python/axtone/frontend/axtone/components/processing-state.tsx",
                        lineNumber: 42,
                        columnNumber: 9
                    }, this),
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                        className: "relative w-full h-full bg-gradient-to-br from-cyan-500 to-blue-600 rounded-lg flex items-center justify-center border border-cyan-400/50",
                        children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                            className: "text-4xl font-bold text-white",
                            children: "♪"
                        }, void 0, false, {
                            fileName: "[project]/codes/python/axtone/frontend/axtone/components/processing-state.tsx",
                            lineNumber: 44,
                            columnNumber: 11
                        }, this)
                    }, void 0, false, {
                        fileName: "[project]/codes/python/axtone/frontend/axtone/components/processing-state.tsx",
                        lineNumber: 43,
                        columnNumber: 9
                    }, this)
                ]
            }, void 0, true, {
                fileName: "[project]/codes/python/axtone/frontend/axtone/components/processing-state.tsx",
                lineNumber: 41,
                columnNumber: 7
            }, this),
            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                className: "text-center space-y-2",
                children: [
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                        className: "text-slate-400 text-sm",
                        children: "Processing"
                    }, void 0, false, {
                        fileName: "[project]/codes/python/axtone/frontend/axtone/components/processing-state.tsx",
                        lineNumber: 50,
                        columnNumber: 9
                    }, this),
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                        className: "text-white font-semibold truncate max-w-xs",
                        children: fileName
                    }, void 0, false, {
                        fileName: "[project]/codes/python/axtone/frontend/axtone/components/processing-state.tsx",
                        lineNumber: 51,
                        columnNumber: 9
                    }, this)
                ]
            }, void 0, true, {
                fileName: "[project]/codes/python/axtone/frontend/axtone/components/processing-state.tsx",
                lineNumber: 49,
                columnNumber: 7
            }, this),
            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                className: "space-y-4 w-full max-w-md",
                children: steps.map((step)=>/*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                        className: "flex items-center gap-4",
                        children: [
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                className: "relative w-8 h-8 flex-shrink-0",
                                children: step.completed ? /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                    className: "w-full h-full rounded-full bg-gradient-to-r from-cyan-500 to-blue-600 flex items-center justify-center",
                                    children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("svg", {
                                        className: "w-5 h-5 text-white",
                                        fill: "none",
                                        stroke: "currentColor",
                                        viewBox: "0 0 24 24",
                                        children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("path", {
                                            strokeLinecap: "round",
                                            strokeLinejoin: "round",
                                            strokeWidth: 3,
                                            d: "M5 13l4 4L19 7"
                                        }, void 0, false, {
                                            fileName: "[project]/codes/python/axtone/frontend/axtone/components/processing-state.tsx",
                                            lineNumber: 62,
                                            columnNumber: 21
                                        }, this)
                                    }, void 0, false, {
                                        fileName: "[project]/codes/python/axtone/frontend/axtone/components/processing-state.tsx",
                                        lineNumber: 61,
                                        columnNumber: 19
                                    }, this)
                                }, void 0, false, {
                                    fileName: "[project]/codes/python/axtone/frontend/axtone/components/processing-state.tsx",
                                    lineNumber: 60,
                                    columnNumber: 17
                                }, this) : /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                    className: "w-full h-full rounded-full border-2 border-slate-700 flex items-center justify-center",
                                    children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$loader$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__$3c$export__default__as__Loader$3e$__["Loader"], {
                                        className: "w-4 h-4 text-cyan-400 animate-spin"
                                    }, void 0, false, {
                                        fileName: "[project]/codes/python/axtone/frontend/axtone/components/processing-state.tsx",
                                        lineNumber: 67,
                                        columnNumber: 19
                                    }, this)
                                }, void 0, false, {
                                    fileName: "[project]/codes/python/axtone/frontend/axtone/components/processing-state.tsx",
                                    lineNumber: 66,
                                    columnNumber: 17
                                }, this)
                            }, void 0, false, {
                                fileName: "[project]/codes/python/axtone/frontend/axtone/components/processing-state.tsx",
                                lineNumber: 58,
                                columnNumber: 13
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                className: `text-sm font-medium ${step.completed ? 'text-slate-300' : 'text-slate-400'}`,
                                children: step.label
                            }, void 0, false, {
                                fileName: "[project]/codes/python/axtone/frontend/axtone/components/processing-state.tsx",
                                lineNumber: 71,
                                columnNumber: 13
                            }, this)
                        ]
                    }, step.id, true, {
                        fileName: "[project]/codes/python/axtone/frontend/axtone/components/processing-state.tsx",
                        lineNumber: 57,
                        columnNumber: 11
                    }, this))
            }, void 0, false, {
                fileName: "[project]/codes/python/axtone/frontend/axtone/components/processing-state.tsx",
                lineNumber: 55,
                columnNumber: 7
            }, this),
            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                className: "text-center",
                children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                    className: "text-slate-500 text-sm",
                    children: "Analyzing audio and building optimal fretboard path..."
                }, void 0, false, {
                    fileName: "[project]/codes/python/axtone/frontend/axtone/components/processing-state.tsx",
                    lineNumber: 80,
                    columnNumber: 9
                }, this)
            }, void 0, false, {
                fileName: "[project]/codes/python/axtone/frontend/axtone/components/processing-state.tsx",
                lineNumber: 79,
                columnNumber: 7
            }, this)
        ]
    }, void 0, true, {
        fileName: "[project]/codes/python/axtone/frontend/axtone/components/processing-state.tsx",
        lineNumber: 39,
        columnNumber: 5
    }, this);
}
}),
"[project]/codes/python/axtone/frontend/axtone/components/result-view.tsx [app-ssr] (ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "default",
    ()=>ResultView
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/codes/python/axtone/frontend/axtone/node_modules/next/dist/server/route-modules/app-page/vendored/ssr/react-jsx-dev-runtime.js [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/codes/python/axtone/frontend/axtone/node_modules/next/dist/server/route-modules/app-page/vendored/ssr/react.js [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$play$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__$3c$export__default__as__Play$3e$__ = __turbopack_context__.i("[project]/codes/python/axtone/frontend/axtone/node_modules/lucide-react/dist/esm/icons/play.js [app-ssr] (ecmascript) <export default as Play>");
var __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$pause$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__$3c$export__default__as__Pause$3e$__ = __turbopack_context__.i("[project]/codes/python/axtone/frontend/axtone/node_modules/lucide-react/dist/esm/icons/pause.js [app-ssr] (ecmascript) <export default as Pause>");
var __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$download$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__$3c$export__default__as__Download$3e$__ = __turbopack_context__.i("[project]/codes/python/axtone/frontend/axtone/node_modules/lucide-react/dist/esm/icons/download.js [app-ssr] (ecmascript) <export default as Download>");
var __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$file$2d$down$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__$3c$export__default__as__FileDown$3e$__ = __turbopack_context__.i("[project]/codes/python/axtone/frontend/axtone/node_modules/lucide-react/dist/esm/icons/file-down.js [app-ssr] (ecmascript) <export default as FileDown>");
var __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$music$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__$3c$export__default__as__Music$3e$__ = __turbopack_context__.i("[project]/codes/python/axtone/frontend/axtone/node_modules/lucide-react/dist/esm/icons/music.js [app-ssr] (ecmascript) <export default as Music>");
var __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$components$2f$ui$2f$button$2e$tsx__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/codes/python/axtone/frontend/axtone/components/ui/button.tsx [app-ssr] (ecmascript)");
'use client';
;
;
;
;
function ResultView({ fileName, isPlaying, onPlayingChange, tablature, stats }) {
    const [copied, setCopied] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useState"])(false);
    const handleCopy = ()=>{
        navigator.clipboard.writeText(tablature);
        setCopied(true);
        setTimeout(()=>setCopied(false), 2000);
    };
    const handleDownloadTab = ()=>{
        const element = document.createElement('a');
        element.setAttribute('href', 'data:text/plain;charset=utf-8,' + encodeURIComponent(tablature));
        element.setAttribute('download', fileName.replace(/\.[^/.]+$/, '') + '_tab.txt');
        element.style.display = 'none';
        document.body.appendChild(element);
        element.click();
        document.body.removeChild(element);
    };
    return /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
        className: "space-y-8",
        children: [
            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                className: "flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4",
                children: [
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                        children: [
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("h2", {
                                className: "text-2xl font-bold text-white",
                                children: "Your Tab"
                            }, void 0, false, {
                                fileName: "[project]/codes/python/axtone/frontend/axtone/components/result-view.tsx",
                                lineNumber: 54,
                                columnNumber: 11
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                                className: "text-slate-400 text-sm mt-1",
                                children: fileName
                            }, void 0, false, {
                                fileName: "[project]/codes/python/axtone/frontend/axtone/components/result-view.tsx",
                                lineNumber: 55,
                                columnNumber: 11
                            }, this)
                        ]
                    }, void 0, true, {
                        fileName: "[project]/codes/python/axtone/frontend/axtone/components/result-view.tsx",
                        lineNumber: 53,
                        columnNumber: 9
                    }, this),
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                        className: "flex gap-2",
                        children: [
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$components$2f$ui$2f$button$2e$tsx__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["Button"], {
                                onClick: ()=>onPlayingChange(!isPlaying),
                                className: "bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-600 hover:to-blue-700 text-white border-0",
                                size: "lg",
                                children: isPlaying ? /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["Fragment"], {
                                    children: [
                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$pause$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__$3c$export__default__as__Pause$3e$__["Pause"], {
                                            className: "w-4 h-4 mr-2"
                                        }, void 0, false, {
                                            fileName: "[project]/codes/python/axtone/frontend/axtone/components/result-view.tsx",
                                            lineNumber: 65,
                                            columnNumber: 17
                                        }, this),
                                        "Pause"
                                    ]
                                }, void 0, true) : /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["Fragment"], {
                                    children: [
                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$play$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__$3c$export__default__as__Play$3e$__["Play"], {
                                            className: "w-4 h-4 mr-2"
                                        }, void 0, false, {
                                            fileName: "[project]/codes/python/axtone/frontend/axtone/components/result-view.tsx",
                                            lineNumber: 70,
                                            columnNumber: 17
                                        }, this),
                                        "Play"
                                    ]
                                }, void 0, true)
                            }, void 0, false, {
                                fileName: "[project]/codes/python/axtone/frontend/axtone/components/result-view.tsx",
                                lineNumber: 58,
                                columnNumber: 11
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$components$2f$ui$2f$button$2e$tsx__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["Button"], {
                                onClick: handleDownloadTab,
                                variant: "outline",
                                size: "lg",
                                className: "border-slate-700 text-slate-300 hover:bg-slate-800 bg-transparent",
                                children: [
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$file$2d$down$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__$3c$export__default__as__FileDown$3e$__["FileDown"], {
                                        className: "w-4 h-4 mr-2"
                                    }, void 0, false, {
                                        fileName: "[project]/codes/python/axtone/frontend/axtone/components/result-view.tsx",
                                        lineNumber: 81,
                                        columnNumber: 13
                                    }, this),
                                    "Download Tab"
                                ]
                            }, void 0, true, {
                                fileName: "[project]/codes/python/axtone/frontend/axtone/components/result-view.tsx",
                                lineNumber: 75,
                                columnNumber: 11
                            }, this)
                        ]
                    }, void 0, true, {
                        fileName: "[project]/codes/python/axtone/frontend/axtone/components/result-view.tsx",
                        lineNumber: 57,
                        columnNumber: 9
                    }, this)
                ]
            }, void 0, true, {
                fileName: "[project]/codes/python/axtone/frontend/axtone/components/result-view.tsx",
                lineNumber: 52,
                columnNumber: 7
            }, this),
            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                className: "grid grid-cols-1 lg:grid-cols-2 gap-8",
                children: [
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                        className: "rounded-xl border border-slate-800 bg-slate-900/50 p-6 backdrop-blur-sm",
                        children: [
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("h3", {
                                className: "text-sm font-semibold text-slate-200 mb-4 flex items-center gap-2",
                                children: [
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$music$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__$3c$export__default__as__Music$3e$__["Music"], {
                                        className: "w-4 h-4 text-cyan-400"
                                    }, void 0, false, {
                                        fileName: "[project]/codes/python/axtone/frontend/axtone/components/result-view.tsx",
                                        lineNumber: 92,
                                        columnNumber: 13
                                    }, this),
                                    "Statistics"
                                ]
                            }, void 0, true, {
                                fileName: "[project]/codes/python/axtone/frontend/axtone/components/result-view.tsx",
                                lineNumber: 91,
                                columnNumber: 11
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                className: "space-y-3",
                                children: [
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                        className: "flex justify-between items-center",
                                        children: [
                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                                className: "text-slate-400 text-sm",
                                                children: "Total Notes"
                                            }, void 0, false, {
                                                fileName: "[project]/codes/python/axtone/frontend/axtone/components/result-view.tsx",
                                                lineNumber: 97,
                                                columnNumber: 15
                                            }, this),
                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                                className: "text-white font-semibold",
                                                children: stats.total_notes
                                            }, void 0, false, {
                                                fileName: "[project]/codes/python/axtone/frontend/axtone/components/result-view.tsx",
                                                lineNumber: 98,
                                                columnNumber: 15
                                            }, this)
                                        ]
                                    }, void 0, true, {
                                        fileName: "[project]/codes/python/axtone/frontend/axtone/components/result-view.tsx",
                                        lineNumber: 96,
                                        columnNumber: 13
                                    }, this),
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                        className: "flex justify-between items-center",
                                        children: [
                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                                className: "text-slate-400 text-sm",
                                                children: "Pitch Range"
                                            }, void 0, false, {
                                                fileName: "[project]/codes/python/axtone/frontend/axtone/components/result-view.tsx",
                                                lineNumber: 101,
                                                columnNumber: 15
                                            }, this),
                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                                className: "text-white font-semibold",
                                                children: [
                                                    stats.pitch_range.min,
                                                    " - ",
                                                    stats.pitch_range.max
                                                ]
                                            }, void 0, true, {
                                                fileName: "[project]/codes/python/axtone/frontend/axtone/components/result-view.tsx",
                                                lineNumber: 102,
                                                columnNumber: 15
                                            }, this)
                                        ]
                                    }, void 0, true, {
                                        fileName: "[project]/codes/python/axtone/frontend/axtone/components/result-view.tsx",
                                        lineNumber: 100,
                                        columnNumber: 13
                                    }, this),
                                    stats.avg_fret_movement !== undefined && /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                        className: "flex justify-between items-center",
                                        children: [
                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                                className: "text-slate-400 text-sm",
                                                children: "Avg Fret Movement"
                                            }, void 0, false, {
                                                fileName: "[project]/codes/python/axtone/frontend/axtone/components/result-view.tsx",
                                                lineNumber: 108,
                                                columnNumber: 17
                                            }, this),
                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                                className: "text-white font-semibold",
                                                children: stats.avg_fret_movement.toFixed(2)
                                            }, void 0, false, {
                                                fileName: "[project]/codes/python/axtone/frontend/axtone/components/result-view.tsx",
                                                lineNumber: 109,
                                                columnNumber: 17
                                            }, this)
                                        ]
                                    }, void 0, true, {
                                        fileName: "[project]/codes/python/axtone/frontend/axtone/components/result-view.tsx",
                                        lineNumber: 107,
                                        columnNumber: 15
                                    }, this),
                                    stats.avg_string_jumps !== undefined && /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                        className: "flex justify-between items-center",
                                        children: [
                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                                className: "text-slate-400 text-sm",
                                                children: "Avg String Jumps"
                                            }, void 0, false, {
                                                fileName: "[project]/codes/python/axtone/frontend/axtone/components/result-view.tsx",
                                                lineNumber: 114,
                                                columnNumber: 17
                                            }, this),
                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                                className: "text-white font-semibold",
                                                children: stats.avg_string_jumps.toFixed(2)
                                            }, void 0, false, {
                                                fileName: "[project]/codes/python/axtone/frontend/axtone/components/result-view.tsx",
                                                lineNumber: 115,
                                                columnNumber: 17
                                            }, this)
                                        ]
                                    }, void 0, true, {
                                        fileName: "[project]/codes/python/axtone/frontend/axtone/components/result-view.tsx",
                                        lineNumber: 113,
                                        columnNumber: 15
                                    }, this)
                                ]
                            }, void 0, true, {
                                fileName: "[project]/codes/python/axtone/frontend/axtone/components/result-view.tsx",
                                lineNumber: 95,
                                columnNumber: 11
                            }, this)
                        ]
                    }, void 0, true, {
                        fileName: "[project]/codes/python/axtone/frontend/axtone/components/result-view.tsx",
                        lineNumber: 90,
                        columnNumber: 9
                    }, this),
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                        className: "rounded-xl border border-slate-800 bg-slate-900/50 p-6 backdrop-blur-sm overflow-x-auto",
                        children: [
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                className: "flex items-center justify-between mb-4",
                                children: [
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("h3", {
                                        className: "text-sm font-semibold text-slate-200 flex items-center gap-2",
                                        children: [
                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$music$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__$3c$export__default__as__Music$3e$__["Music"], {
                                                className: "w-4 h-4 text-cyan-400"
                                            }, void 0, false, {
                                                fileName: "[project]/codes/python/axtone/frontend/axtone/components/result-view.tsx",
                                                lineNumber: 125,
                                                columnNumber: 15
                                            }, this),
                                            "Tablature"
                                        ]
                                    }, void 0, true, {
                                        fileName: "[project]/codes/python/axtone/frontend/axtone/components/result-view.tsx",
                                        lineNumber: 124,
                                        columnNumber: 13
                                    }, this),
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$components$2f$ui$2f$button$2e$tsx__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["Button"], {
                                        onClick: handleCopy,
                                        variant: "ghost",
                                        size: "sm",
                                        className: "text-slate-400 hover:text-slate-200 hover:bg-slate-800 text-xs",
                                        children: copied ? 'Copied!' : 'Copy'
                                    }, void 0, false, {
                                        fileName: "[project]/codes/python/axtone/frontend/axtone/components/result-view.tsx",
                                        lineNumber: 128,
                                        columnNumber: 13
                                    }, this)
                                ]
                            }, void 0, true, {
                                fileName: "[project]/codes/python/axtone/frontend/axtone/components/result-view.tsx",
                                lineNumber: 123,
                                columnNumber: 11
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                className: "font-mono text-sm text-slate-300 leading-relaxed whitespace-pre-wrap break-words max-w-full",
                                children: tablature
                            }, void 0, false, {
                                fileName: "[project]/codes/python/axtone/frontend/axtone/components/result-view.tsx",
                                lineNumber: 137,
                                columnNumber: 11
                            }, this)
                        ]
                    }, void 0, true, {
                        fileName: "[project]/codes/python/axtone/frontend/axtone/components/result-view.tsx",
                        lineNumber: 122,
                        columnNumber: 9
                    }, this)
                ]
            }, void 0, true, {
                fileName: "[project]/codes/python/axtone/frontend/axtone/components/result-view.tsx",
                lineNumber: 88,
                columnNumber: 7
            }, this),
            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                className: "rounded-xl border border-slate-800 bg-slate-900/50 p-6 backdrop-blur-sm",
                children: [
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("h3", {
                        className: "text-sm font-semibold text-slate-200 mb-4",
                        children: "Export Options"
                    }, void 0, false, {
                        fileName: "[project]/codes/python/axtone/frontend/axtone/components/result-view.tsx",
                        lineNumber: 145,
                        columnNumber: 9
                    }, this),
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                        className: "flex flex-col sm:flex-row gap-3",
                        children: [
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$components$2f$ui$2f$button$2e$tsx__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["Button"], {
                                onClick: handleDownloadTab,
                                className: "flex-1 bg-slate-800 hover:bg-slate-700 text-slate-100",
                                children: [
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$file$2d$down$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__$3c$export__default__as__FileDown$3e$__["FileDown"], {
                                        className: "w-4 h-4 mr-2"
                                    }, void 0, false, {
                                        fileName: "[project]/codes/python/axtone/frontend/axtone/components/result-view.tsx",
                                        lineNumber: 151,
                                        columnNumber: 13
                                    }, this),
                                    "Download as .txt"
                                ]
                            }, void 0, true, {
                                fileName: "[project]/codes/python/axtone/frontend/axtone/components/result-view.tsx",
                                lineNumber: 147,
                                columnNumber: 11
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$components$2f$ui$2f$button$2e$tsx__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["Button"], {
                                className: "flex-1 bg-slate-800 hover:bg-slate-700 text-slate-100",
                                children: [
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$download$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__$3c$export__default__as__Download$3e$__["Download"], {
                                        className: "w-4 h-4 mr-2"
                                    }, void 0, false, {
                                        fileName: "[project]/codes/python/axtone/frontend/axtone/components/result-view.tsx",
                                        lineNumber: 157,
                                        columnNumber: 13
                                    }, this),
                                    "Download as MIDI"
                                ]
                            }, void 0, true, {
                                fileName: "[project]/codes/python/axtone/frontend/axtone/components/result-view.tsx",
                                lineNumber: 154,
                                columnNumber: 11
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$components$2f$ui$2f$button$2e$tsx__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["Button"], {
                                variant: "outline",
                                className: "flex-1 border-slate-700 text-slate-300 hover:bg-slate-800 bg-transparent",
                                children: "Share Link"
                            }, void 0, false, {
                                fileName: "[project]/codes/python/axtone/frontend/axtone/components/result-view.tsx",
                                lineNumber: 160,
                                columnNumber: 11
                            }, this)
                        ]
                    }, void 0, true, {
                        fileName: "[project]/codes/python/axtone/frontend/axtone/components/result-view.tsx",
                        lineNumber: 146,
                        columnNumber: 9
                    }, this)
                ]
            }, void 0, true, {
                fileName: "[project]/codes/python/axtone/frontend/axtone/components/result-view.tsx",
                lineNumber: 144,
                columnNumber: 7
            }, this)
        ]
    }, void 0, true, {
        fileName: "[project]/codes/python/axtone/frontend/axtone/components/result-view.tsx",
        lineNumber: 50,
        columnNumber: 5
    }, this);
}
}),
"[project]/codes/python/axtone/frontend/axtone/components/interactive-fretboard.tsx [app-ssr] (ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "default",
    ()=>InteractiveFretboard
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/codes/python/axtone/frontend/axtone/node_modules/next/dist/server/route-modules/app-page/vendored/ssr/react-jsx-dev-runtime.js [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/codes/python/axtone/frontend/axtone/node_modules/next/dist/server/route-modules/app-page/vendored/ssr/react.js [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$guitar$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__$3c$export__default__as__Guitar$3e$__ = __turbopack_context__.i("[project]/codes/python/axtone/frontend/axtone/node_modules/lucide-react/dist/esm/icons/guitar.js [app-ssr] (ecmascript) <export default as Guitar>");
'use client';
;
;
;
const STRINGS = [
    {
        name: 'E',
        color: '#ef4444',
        frequency: 82.41
    },
    {
        name: 'A',
        color: '#f97316',
        frequency: 110.0
    },
    {
        name: 'D',
        color: '#eab308',
        frequency: 146.83
    },
    {
        name: 'G',
        color: '#22c55e',
        frequency: 196.0
    },
    {
        name: 'B',
        color: '#06b6d4',
        frequency: 246.94
    },
    {
        name: 'e',
        color: '#8b5cf6',
        frequency: 329.63
    }
];
const FRETS = 24;
function InteractiveFretboard({ isPlaying }) {
    const canvasRef = (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useRef"])(null);
    const [highlightedFret, setHighlightedFret] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useState"])(null);
    (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useEffect"])(()=>{
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;
        const padding = 40;
        const width = canvas.width;
        const height = canvas.height;
        const stringSpacing = (height - 2 * padding) / (STRINGS.length - 1);
        const fretSpacing = (width - 2 * padding) / FRETS;
        // Clear canvas
        ctx.fillStyle = '#0f172a';
        ctx.fillRect(0, 0, width, height);
        // Draw frets
        ctx.strokeStyle = '#334155';
        ctx.lineWidth = 1;
        for(let i = 0; i <= FRETS; i++){
            const x = padding + i * fretSpacing;
            ctx.beginPath();
            ctx.moveTo(x, padding);
            ctx.lineTo(x, height - padding);
            ctx.stroke();
        }
        // Draw strings
        STRINGS.forEach((string, stringIndex)=>{
            const y = padding + stringIndex * stringSpacing;
            // String line
            ctx.strokeStyle = string.color;
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(padding, y);
            ctx.lineTo(width - padding, y);
            ctx.stroke();
            // String label
            ctx.fillStyle = '#cbd5e1';
            ctx.font = '14px monospace';
            ctx.textAlign = 'right';
            ctx.fillText(string.name, padding - 15, y + 5);
            // Draw fret dots
            ctx.fillStyle = '#64748b';
            for(let fret = 0; fret <= FRETS; fret++){
                const x = padding + fret * fretSpacing;
                const dotSize = 3;
                // Highlight if this is the current playing note
                if (isPlaying && highlightedFret && stringIndex === highlightedFret.string && fret === highlightedFret.fret) {
                    ctx.fillStyle = string.color;
                    ctx.fillRect(x - 5, y - 5, 10, 10);
                }
            }
        });
        // Animate highlighted note when playing
        if (isPlaying) {
            const time = Date.now() / 1000;
            const notePattern = [
                {
                    string: 1,
                    fret: 2
                },
                {
                    string: 2,
                    fret: 3
                },
                {
                    string: 3,
                    fret: 2
                },
                {
                    string: 1,
                    fret: 5
                },
                {
                    string: 2,
                    fret: 7
                },
                {
                    string: 3,
                    fret: 5
                }
            ];
            const index = Math.floor(time * 2) % notePattern.length;
            setHighlightedFret(notePattern[index]);
        }
    }, [
        isPlaying,
        highlightedFret
    ]);
    return /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
        className: "rounded-xl border border-slate-800 bg-slate-900/50 p-6 backdrop-blur-sm",
        children: [
            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("h3", {
                className: "text-sm font-semibold text-slate-200 mb-4 flex items-center gap-2",
                children: [
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$guitar$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__$3c$export__default__as__Guitar$3e$__["Guitar"], {
                        className: "w-4 h-4 text-cyan-400"
                    }, void 0, false, {
                        fileName: "[project]/codes/python/axtone/frontend/axtone/components/interactive-fretboard.tsx",
                        lineNumber: 111,
                        columnNumber: 9
                    }, this),
                    "Interactive Fretboard"
                ]
            }, void 0, true, {
                fileName: "[project]/codes/python/axtone/frontend/axtone/components/interactive-fretboard.tsx",
                lineNumber: 110,
                columnNumber: 7
            }, this),
            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                className: "overflow-x-auto",
                children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("canvas", {
                    ref: canvasRef,
                    width: 800,
                    height: 280,
                    className: "w-full rounded-lg bg-slate-950 border border-slate-800 min-w-full"
                }, void 0, false, {
                    fileName: "[project]/codes/python/axtone/frontend/axtone/components/interactive-fretboard.tsx",
                    lineNumber: 115,
                    columnNumber: 9
                }, this)
            }, void 0, false, {
                fileName: "[project]/codes/python/axtone/frontend/axtone/components/interactive-fretboard.tsx",
                lineNumber: 114,
                columnNumber: 7
            }, this),
            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                className: "text-xs text-slate-500 mt-3",
                children: isPlaying ? '♪ Playing - watch as notes highlight on the fretboard' : 'Press play to see notes highlighted on the fretboard'
            }, void 0, false, {
                fileName: "[project]/codes/python/axtone/frontend/axtone/components/interactive-fretboard.tsx",
                lineNumber: 122,
                columnNumber: 7
            }, this)
        ]
    }, void 0, true, {
        fileName: "[project]/codes/python/axtone/frontend/axtone/components/interactive-fretboard.tsx",
        lineNumber: 109,
        columnNumber: 5
    }, this);
}
}),
"[project]/codes/python/axtone/frontend/axtone/app/page.tsx [app-ssr] (ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "default",
    ()=>Home
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/codes/python/axtone/frontend/axtone/node_modules/next/dist/server/route-modules/app-page/vendored/ssr/react-jsx-dev-runtime.js [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/codes/python/axtone/frontend/axtone/node_modules/next/dist/server/route-modules/app-page/vendored/ssr/react.js [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$components$2f$file$2d$uploader$2e$tsx__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/codes/python/axtone/frontend/axtone/components/file-uploader.tsx [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$components$2f$processing$2d$state$2e$tsx__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/codes/python/axtone/frontend/axtone/components/processing-state.tsx [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$components$2f$result$2d$view$2e$tsx__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/codes/python/axtone/frontend/axtone/components/result-view.tsx [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$components$2f$interactive$2d$fretboard$2e$tsx__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/codes/python/axtone/frontend/axtone/components/interactive-fretboard.tsx [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$components$2f$ui$2f$button$2e$tsx__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/codes/python/axtone/frontend/axtone/components/ui/button.tsx [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$music$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__$3c$export__default__as__Music$3e$__ = __turbopack_context__.i("[project]/codes/python/axtone/frontend/axtone/node_modules/lucide-react/dist/esm/icons/music.js [app-ssr] (ecmascript) <export default as Music>");
'use client';
;
;
;
;
;
;
;
;
function Home() {
    const [state, setState] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useState"])('idle');
    const [fileName, setFileName] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useState"])('');
    const [isPlaying, setIsPlaying] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useState"])(false);
    const [result, setResult] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useState"])(null);
    const [error, setError] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useState"])('');
    const [tuning, setTuning] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useState"])('standard');
    const [method, setMethod] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useState"])('pyin');
    const handleFileUpload = async (file)=>{
        setFileName(file.name);
        setState('processing');
        setError('');
        try {
            // Create FormData
            const formData = new FormData();
            formData.append('file', file);
            formData.append('method', method);
            formData.append('tuning', tuning);
            formData.append('min_duration', '0.1');
            formData.append('detailed', 'false');
            formData.append('preprocess', 'false');
            // Call the Next.js API route (which forwards to Python)
            const response = await fetch('/api/convert', {
                method: 'POST',
                body: formData
            });
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Failed to convert audio');
            }
            const data = await response.json();
            setResult(data);
            setState('result');
        } catch (err) {
            console.error('Conversion error:', err);
            setError(err instanceof Error ? err.message : 'An error occurred during conversion');
            setState('error');
        }
    };
    const handleReset = ()=>{
        setState('idle');
        setFileName('');
        setIsPlaying(false);
        setResult(null);
        setError('');
    };
    return /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
        className: "min-h-screen bg-gradient-to-b from-slate-950 via-slate-900 to-slate-950",
        children: [
            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("header", {
                className: "border-b border-slate-800 bg-slate-950/50 backdrop-blur-sm sticky top-0 z-50",
                children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                    className: "max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex items-center justify-between",
                    children: [
                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                            className: "flex items-center gap-3",
                            children: [
                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                    className: "w-10 h-10 bg-gradient-to-br from-cyan-400 to-blue-600 rounded-lg flex items-center justify-center",
                                    children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$music$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__$3c$export__default__as__Music$3e$__["Music"], {
                                        className: "w-6 h-6 text-white"
                                    }, void 0, false, {
                                        fileName: "[project]/codes/python/axtone/frontend/axtone/app/page.tsx",
                                        lineNumber: 90,
                                        columnNumber: 15
                                    }, this)
                                }, void 0, false, {
                                    fileName: "[project]/codes/python/axtone/frontend/axtone/app/page.tsx",
                                    lineNumber: 89,
                                    columnNumber: 13
                                }, this),
                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                    children: [
                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("h1", {
                                            className: "text-xl font-bold text-white",
                                            children: "Axtone"
                                        }, void 0, false, {
                                            fileName: "[project]/codes/python/axtone/frontend/axtone/app/page.tsx",
                                            lineNumber: 93,
                                            columnNumber: 15
                                        }, this),
                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                                            className: "text-xs text-slate-400",
                                            children: "AI Vocal to Guitar Tab"
                                        }, void 0, false, {
                                            fileName: "[project]/codes/python/axtone/frontend/axtone/app/page.tsx",
                                            lineNumber: 94,
                                            columnNumber: 15
                                        }, this)
                                    ]
                                }, void 0, true, {
                                    fileName: "[project]/codes/python/axtone/frontend/axtone/app/page.tsx",
                                    lineNumber: 92,
                                    columnNumber: 13
                                }, this)
                            ]
                        }, void 0, true, {
                            fileName: "[project]/codes/python/axtone/frontend/axtone/app/page.tsx",
                            lineNumber: 88,
                            columnNumber: 11
                        }, this),
                        state !== 'idle' && /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$components$2f$ui$2f$button$2e$tsx__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["Button"], {
                            onClick: handleReset,
                            variant: "outline",
                            size: "sm",
                            className: "border-slate-700 text-slate-300 hover:bg-slate-800 bg-transparent",
                            children: "New Upload"
                        }, void 0, false, {
                            fileName: "[project]/codes/python/axtone/frontend/axtone/app/page.tsx",
                            lineNumber: 98,
                            columnNumber: 13
                        }, this)
                    ]
                }, void 0, true, {
                    fileName: "[project]/codes/python/axtone/frontend/axtone/app/page.tsx",
                    lineNumber: 87,
                    columnNumber: 9
                }, this)
            }, void 0, false, {
                fileName: "[project]/codes/python/axtone/frontend/axtone/app/page.tsx",
                lineNumber: 86,
                columnNumber: 7
            }, this),
            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("main", {
                className: "max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 sm:py-12",
                children: [
                    state === 'idle' && /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                        className: "space-y-12",
                        children: [
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                className: "text-center space-y-4 py-8",
                                children: [
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("h2", {
                                        className: "text-4xl sm:text-5xl font-bold text-white text-balance",
                                        children: [
                                            "Convert Vocals to ",
                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                                className: "bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent",
                                                children: "Guitar Tabs"
                                            }, void 0, false, {
                                                fileName: "[project]/codes/python/axtone/frontend/axtone/app/page.tsx",
                                                lineNumber: 117,
                                                columnNumber: 35
                                            }, this)
                                        ]
                                    }, void 0, true, {
                                        fileName: "[project]/codes/python/axtone/frontend/axtone/app/page.tsx",
                                        lineNumber: 116,
                                        columnNumber: 15
                                    }, this),
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                                        className: "text-lg text-slate-400 max-w-2xl mx-auto",
                                        children: "Upload a vocal recording and our AI will analyze the melody and generate accurate guitar tablature in seconds."
                                    }, void 0, false, {
                                        fileName: "[project]/codes/python/axtone/frontend/axtone/app/page.tsx",
                                        lineNumber: 119,
                                        columnNumber: 15
                                    }, this)
                                ]
                            }, void 0, true, {
                                fileName: "[project]/codes/python/axtone/frontend/axtone/app/page.tsx",
                                lineNumber: 115,
                                columnNumber: 13
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$components$2f$file$2d$uploader$2e$tsx__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["default"], {
                                onFileSelect: handleFileUpload
                            }, void 0, false, {
                                fileName: "[project]/codes/python/axtone/frontend/axtone/app/page.tsx",
                                lineNumber: 125,
                                columnNumber: 13
                            }, this)
                        ]
                    }, void 0, true, {
                        fileName: "[project]/codes/python/axtone/frontend/axtone/app/page.tsx",
                        lineNumber: 113,
                        columnNumber: 11
                    }, this),
                    state === 'processing' && /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$components$2f$processing$2d$state$2e$tsx__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["default"], {
                        fileName: fileName
                    }, void 0, false, {
                        fileName: "[project]/codes/python/axtone/frontend/axtone/app/page.tsx",
                        lineNumber: 130,
                        columnNumber: 11
                    }, this),
                    state === 'error' && /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                        className: "max-w-2xl mx-auto",
                        children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                            className: "rounded-xl border border-red-800 bg-red-900/20 p-6 backdrop-blur-sm",
                            children: [
                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("h3", {
                                    className: "text-xl font-bold text-red-400 mb-2",
                                    children: "Conversion Failed"
                                }, void 0, false, {
                                    fileName: "[project]/codes/python/axtone/frontend/axtone/app/page.tsx",
                                    lineNumber: 136,
                                    columnNumber: 15
                                }, this),
                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                                    className: "text-red-300 mb-4",
                                    children: error
                                }, void 0, false, {
                                    fileName: "[project]/codes/python/axtone/frontend/axtone/app/page.tsx",
                                    lineNumber: 137,
                                    columnNumber: 15
                                }, this),
                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$components$2f$ui$2f$button$2e$tsx__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["Button"], {
                                    onClick: handleReset,
                                    className: "bg-red-600 hover:bg-red-700 text-white",
                                    children: "Try Again"
                                }, void 0, false, {
                                    fileName: "[project]/codes/python/axtone/frontend/axtone/app/page.tsx",
                                    lineNumber: 138,
                                    columnNumber: 15
                                }, this)
                            ]
                        }, void 0, true, {
                            fileName: "[project]/codes/python/axtone/frontend/axtone/app/page.tsx",
                            lineNumber: 135,
                            columnNumber: 13
                        }, this)
                    }, void 0, false, {
                        fileName: "[project]/codes/python/axtone/frontend/axtone/app/page.tsx",
                        lineNumber: 134,
                        columnNumber: 11
                    }, this),
                    state === 'result' && result && /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["Fragment"], {
                        children: [
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$components$2f$result$2d$view$2e$tsx__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["default"], {
                                fileName: fileName,
                                isPlaying: isPlaying,
                                onPlayingChange: setIsPlaying,
                                tablature: result.tablature,
                                stats: result.stats
                            }, void 0, false, {
                                fileName: "[project]/codes/python/axtone/frontend/axtone/app/page.tsx",
                                lineNumber: 150,
                                columnNumber: 13
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                className: "mt-12",
                                children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$codes$2f$python$2f$axtone$2f$frontend$2f$axtone$2f$components$2f$interactive$2d$fretboard$2e$tsx__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["default"], {
                                    isPlaying: isPlaying
                                }, void 0, false, {
                                    fileName: "[project]/codes/python/axtone/frontend/axtone/app/page.tsx",
                                    lineNumber: 158,
                                    columnNumber: 15
                                }, this)
                            }, void 0, false, {
                                fileName: "[project]/codes/python/axtone/frontend/axtone/app/page.tsx",
                                lineNumber: 157,
                                columnNumber: 13
                            }, this)
                        ]
                    }, void 0, true)
                ]
            }, void 0, true, {
                fileName: "[project]/codes/python/axtone/frontend/axtone/app/page.tsx",
                lineNumber: 111,
                columnNumber: 7
            }, this)
        ]
    }, void 0, true, {
        fileName: "[project]/codes/python/axtone/frontend/axtone/app/page.tsx",
        lineNumber: 84,
        columnNumber: 5
    }, this);
}
}),
];

//# sourceMappingURL=codes_python_axtone_frontend_axtone_45ece299._.js.map