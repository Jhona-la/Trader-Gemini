import React from 'react';

import { Stack, Text } from '@grafana/ui';
import { Trans, t } from 'app/core/internationalization';

import { Banner } from './Banner';
import { NotificationBannerSpec } from './api/types';

export type BannerPreviewProps = Pick<NotificationBannerSpec, 'message' | 'variant'>;

export function BannerPreview({ message, variant }: BannerPreviewProps) {
  return (
    <Stack direction={'column'} width={'100%'} gap={2}>
      <Text variant="h3">
        <Trans i18nKey="banner.preview.title">Preview</Trans>
      </Text>
      <Banner
        message={message || t('banner.preview.message.default', 'Update this preview by modifying the message field.')}
        variant={variant}
      />
    </Stack>
  );
}
