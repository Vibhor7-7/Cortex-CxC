import { chromium } from 'playwright';
const times = process.argv.slice(2).map(Number);
const browser = await chromium.launch({ args: (process.env.GLARGS||'--use-angle=metal --enable-gpu --ignore-gpu-blocklist').split(' ') });
const page = await browser.newPage({ viewport: { width: 1440, height: 810 }, deviceScaleFactor: 4/3 });
page.on('console', m => { if (m.type()==='error') console.log('ERR', m.text()); });
page.on('pageerror', e => console.log('PAGEERR', e.message));
await page.goto('http://localhost:8765/brag-output/work/comp.html');
await page.waitForFunction(() => window.READY === true, null, { timeout: 30000 });
for (const t of times) {
  await page.evaluate(t => window.seek(t), t);
  await page.screenshot({ path: `stills/t${t.toFixed(2)}.jpg`, type: 'jpeg', quality: 85 });
}
await browser.close();
