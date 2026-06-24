import { css } from '@emotion/css';
import React from 'react';

import { dateMath, dateTime, GrafanaTheme2 } from '@grafana/data';
import { SceneTimeRangeState } from '@grafana/scenes';
import { useTheme2 } from '@grafana/ui';

interface HeaderProps {
  title: string;
  timeRange: SceneTimeRangeState;
  scaleFactor: number;
}

function ReportHeader({ title, timeRange, scaleFactor }: HeaderProps) {
  const theme = useTheme2();
  const styles = getStyles(theme, scaleFactor);
  const dateDisplayFormat = 'YYYY-MM-DD HH:mm:ss Z';

  return (
    <div>
      <div className={styles.subHeader}>
        <div>
          {title} - Generated on {dateMath.parse(dateTime(Date.now()))?.format('dddd, MMM DD YYYY')}
        </div>
        <div className={styles.timeRangeBlock}>
          <div>
            Data time range:{' '}
            {dateMath
              .parse(timeRange.from, true, timeRange.timeZone, timeRange.fiscalYearStartMonth)
              ?.format(dateDisplayFormat)}
            <br />
            {dateMath
              .parse(timeRange.to, true, timeRange.timeZone, timeRange.fiscalYearStartMonth)
              ?.format(dateDisplayFormat)}
          </div>
        </div>
      </div>
      <div className={styles.dashboardTitle}>{title}</div>
      <hr className={styles.divider} />
    </div>
  );
}

function getStyles(theme: GrafanaTheme2, scaleFactor: number) {
  return {
    subHeader: css({
      label: 'subheader',
      display: 'flex',
      fontWeight: theme.typography.fontWeightMedium,
      // TODO: We should use value from theme
      fontSize: `${0.85 * scaleFactor}rem`,
    }),
    timeRangeBlock: css({
      label: 'time-range-block',
      display: 'flex',
      justifyContent: 'flex-end',
      textAlign: 'right',
      flexGrow: 1,
    }),
    dashboardTitle: css({
      label: 'dashboard-title',
      // TODO: We should use value from theme
      fontSize: `${1.7 * scaleFactor}rem`,
      fontWeight: theme.typography.fontWeightMedium,
    }),
    divider: css({
      marginTop: 0,
      marginBottom: theme.spacing(2 * scaleFactor),
    }),
  };
}

export default ReportHeader;
