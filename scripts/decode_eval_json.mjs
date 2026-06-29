import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const src = path.join(__dirname, '..', 'agent-tools', 'fcf51e2e-ab5b-47ab-a3e2-8a4002f869f9.txt');
const out = path.join(__dirname, 'evaluate_model_weighted.py');

const data = JSON.parse(fs.readFileSync(src, 'utf8'));
let t = data.resource.text;

// Run build_weighted_eval.mjs in same directory for full patch list.
// This helper decodes branch evaluate_model JSON for local editing.
fs.writeFileSync(out, t, 'utf8');
console.log('Wrote', out, t.length);
