import React from 'react';
import { useParams } from 'react-router-dom';

import { Page } from 'app/core/components/Page/Page';
import { useDataSourceSettingsNav } from 'app/features/connections/hooks/useDataSourceSettingsNav';

import DataSourcePermissions from '../permissions/DataSourcePermissions';

// The RBAC version of the data source permissions page
export const DataSourcePermissionsPage = () => {
  const { uid } = useParams<{ uid: string }>();
  const { navId, pageNav } = useDataSourceSettingsNav('permissions');

  return (
    <Page navId={navId} pageNav={pageNav}>
      <Page.Contents>
        <DataSourcePermissions uid={uid} />
      </Page.Contents>
    </Page>
  );
};
