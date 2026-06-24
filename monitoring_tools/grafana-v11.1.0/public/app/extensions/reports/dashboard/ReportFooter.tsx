import { css } from '@emotion/css';
import React from 'react';

import { GrafanaTheme2 } from '@grafana/data';
import { useTheme2 } from '@grafana/ui';

interface FooterProps {
  scaleFactor: number;
  // pageNb: number;
  // totalPageCount: number;
}

function ReportFooter({ scaleFactor }: FooterProps) {
  const theme = useTheme2();
  const styles = getStyles(theme, scaleFactor);

  return (
    <div className={styles.footer}>
      {/* TODO: This doesn't work - This should be fixed or page count should be added in the backend
      <div className={styles.pageCount}>
        Page {pageNb}/{totalPageCount}
      </div>
      */}
      <div className={styles.logo}>
        <img className={styles.logoImg} src="public/app/extensions/reports/img/grafana-logo.png" alt="Report logo" />
      </div>
    </div>
  );
}

function getStyles(theme: GrafanaTheme2, scaleFactor: number) {
  return {
    footer: css({
      position: 'fixed',
      bottom: 0,
      height: theme.spacing(4 * scaleFactor),
      width: '100%',
    }),
    pageCount: css({
      height: '100%',
      display: 'flex',
      fontWeight: theme.typography.fontWeightMedium,
      fontSize: theme.typography.bodySmall.fontSize,
      justifyContent: 'center',
      alignItems: 'center',
    }),
    logo: css({
      position: 'absolute',
      top: theme.spacing(0.5 * scaleFactor),
      right: theme.spacing(2 + 0.5 * scaleFactor),
    }),
    logoImg: css({
      maxWidth: theme.spacing(3 * scaleFactor),
      maxHeight: theme.spacing(3 * scaleFactor),
    }),
  };
}

export default ReportFooter;
