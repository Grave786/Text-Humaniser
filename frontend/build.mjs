import path from "node:path";
import { fileURLToPath } from "node:url";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { minify } from "terser";
import JavaScriptObfuscator from "javascript-obfuscator";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const entries = [
  { name: "admin", sources: ["js/config.js", "js/admin.js"] },
  { name: "humanizer", sources: ["js/config.js", "js/humanizer.js"] },
  { name: "landing", sources: ["js/config.js", "js/landing.js"] },
  { name: "login", sources: ["js/config.js", "js/login.js"] },
];

const buildOne = async ({ name, sources }) => {
  const sourceTexts = await Promise.all(
    sources.map((rel) => readFile(path.join(__dirname, rel), "utf8")),
  );

  const combined = sourceTexts.join("\n;\n");

  const terserResult = await minify(combined, {
    compress: {
      passes: 2,
      drop_console: false,
      drop_debugger: true,
    },
    mangle: true,
    toplevel: true,
    format: { comments: false },
  });

  if (!terserResult.code) {
    throw new Error(`Terser did not return code for ${name}.`);
  }

  const obfuscation = JavaScriptObfuscator.obfuscate(terserResult.code, {
    compact: true,
    controlFlowFlattening: false,
    deadCodeInjection: false,
    debugProtection: false,
    disableConsoleOutput: false,
    identifierNamesGenerator: "hexadecimal",
    renameGlobals: false,
    rotateStringArray: true,
    selfDefending: false,
    stringArray: true,
    stringArrayEncoding: ["none"],
    stringArrayThreshold: 0.75,
    unicodeEscapeSequence: false,
  });

  const outDir = path.join(__dirname, "dist");
  await mkdir(outDir, { recursive: true });
  const outPath = path.join(outDir, `${name}.bundle.min.js`);
  await writeFile(outPath, obfuscation.getObfuscatedCode(), "utf8");
  return outPath;
};

  const main = async () => {
  const built = [];
  for (const entry of entries) {
    built.push(await buildOne(entry));
  }
  // Keep output minimal for CI logs.
  process.stdout.write(`Built ${built.length} bundles into frontend/dist\n`);
};

await main();
