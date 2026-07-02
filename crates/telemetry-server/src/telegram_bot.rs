use reqwest::Client;
use serde_json::json;
use std::env;

pub struct TelegramBot {
    client: Client,
    token: String,
    chat_id: String,
}

impl TelegramBot {
    pub fn new() -> Option<Self> {
        let token = env::var("TELEGRAM_BOT_TOKEN").ok()?;
        let chat_id = env::var("TELEGRAM_CHAT_ID").ok()?;

        Some(Self {
            client: Client::new(),
            token,
            chat_id,
        })
    }

    /// Envía un mensaje de texto al chat configurado.
    pub async fn send_message(&self, text: &str) -> Result<(), Box<dyn std::error::Error>> {
        let url = format!("https://api.telegram.org/bot{}/sendMessage", self.token);
        
        let payload = json!({
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "MarkdownV2"
        });

        self.client.post(&url)
            .json(&payload)
            .send()
            .await?;

        Ok(())
    }

    /// Envía una alerta de urgencia formateada
    pub async fn send_alert(&self, title: &str, message: &str) -> Result<(), Box<dyn std::error::Error>> {
        // Escapar caracteres para MarkdownV2
        let escaped_title = Self::escape_markdown(title);
        let escaped_message = Self::escape_markdown(message);

        let formatted = format!("🚨 *{}*\n\n{}", escaped_title, escaped_message);
        self.send_message(&formatted).await
    }

    fn escape_markdown(text: &str) -> String {
        // En MarkdownV2 de Telegram hay que escapar varios caracteres especiales
        let specials = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!'];
        let mut escaped = String::with_capacity(text.len());
        
        for c in text.chars() {
            if specials.contains(&c) {
                escaped.push('\\');
            }
            escaped.push(c);
        }
        escaped
    }
}
