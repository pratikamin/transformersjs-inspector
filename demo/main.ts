import { VERSION } from '../src/index';

const out = document.querySelector<HTMLOutputElement>('[data-version]');
if (out) out.textContent = VERSION;
