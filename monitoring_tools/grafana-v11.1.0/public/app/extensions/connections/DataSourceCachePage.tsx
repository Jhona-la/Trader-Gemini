import React from 'react';
import { useParams } from 'react-router-dom';

import { Page } from 'app/core/components/Page/Page';
import { useDataSourceSettingsNav } from 'app/features/connections/hooks/useDataSourceSettingsNav';

import DataSourceCache from '../caching/DataSourceCache';

export const DataSourceCachePage = () => {
  const { uid } = useParams<{ uid: string }>();
  const { navId, pageNav } = useDataSourceSettingsNav('cache');

  return (
    <Page navId={navId} pageNav={pageNav}>
      <Page.Contents>
        <DataSourceCache uid={uid} />
      </Page.Contents>
    </Page>
  );
};
