import { chromium } from 'playwright';
import fs from 'fs';
const FPS = 30, DUR = 23.0, NF = Math.round(FPS * DUR);
fs.mkdirSync('frames', { recursive: true });
const browser = await chromium.launch({ args: ['--use-angle=metal','--enable-gpu','--ignore-gpu-blocklist'] });
const page = await browser.newPage({ viewport: { width: 1440, height: 810 }, deviceScaleFactor: 4/3 });
page.on('pageerror', e => console.log('PAGEERR', e.message));
await page.goto('http://localhost:8765/brag-output/work/comp.html');
await page.waitForFunction(() => window.READY === true);
const F0 = +(process.argv[2] ?? 0), F1 = +(process.argv[3] ?? NF);
for (let f = F0; f < F1; f++) {
  await page.evaluate(t => window.seek(t), f / FPS);
  await page.screenshot({ path: `frames/f${String(f).padStart(4,'0')}.jpg`, type: 'jpeg', quality: 95 });
  if (f % 100 === 0) console.log('frame', f);
}
await browser.close();
console.log('done', NF);
