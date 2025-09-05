import { Html, Head, Main, NextScript } from "next/document";

export default function Document() {
  return (
    <Html lang="en" className="dark" style={{ backgroundColor: '#000000' }}>
      <Head>
        <link rel="icon" href="/icon.ico" />
        <link rel="shortcut icon" href="/icon.ico" />
        <meta name="description" content="Advanced signals from top crypto wallets - Track elite traders and discover profitable opportunities" />
        <style dangerouslySetInnerHTML={{
          __html: `
            html, body {
              background-color: #000000 !important;
              margin: 0 !important;
              padding: 0 !important;
            }
          `
        }} />
      </Head>
      <body className="antialiased" style={{ backgroundColor: '#000000', margin: 0, padding: 0 }}>
        <Main />
        <NextScript />
      </body>
    </Html>
  );
}
