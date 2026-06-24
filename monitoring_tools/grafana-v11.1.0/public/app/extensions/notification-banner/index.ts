import { addPageBanner } from 'app/AppWrapper';

import { TopBanner } from './TopBanner';

export async function initNotificationBanners() {
  addPageBanner(TopBanner);
}
