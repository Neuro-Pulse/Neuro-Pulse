# Contributing to NeuroPulse

Thank you for your interest in contributing to NeuroPulse. This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. Professional behavior and respectful communication are expected from all contributors.

## Development Process

We follow a structured development process to maintain code quality and system stability.

### Prerequisites

- Node.js >= 18.0.0
- npm >= 9.0.0
- Docker and Docker Compose
- Git
- Solana CLI tools
- PostgreSQL, Redis, MongoDB (for local development)

### Setup Development Environment

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/neuropulse.git
   cd neuropulse
   ```

3. Install dependencies:
   ```bash
   npm install
   ```

4. Copy environment configuration:
   ```bash
   cp .env.example .env
   ```

5. Configure your local environment variables in `.env`

6. Run database migrations:
   ```bash
   npm run migrate:dev
   ```

7. Start development server:
   ```bash
   npm run dev
   ```

## Contribution Workflow

### Branch Naming Convention

- `feature/` - New features
- `fix/` - Bug fixes
- `refactor/` - Code refactoring
- `docs/` - Documentation updates
- `test/` - Test additions or modifications
- `perf/` - Performance improvements

### Commit Message Format

We follow the Conventional Commits specification:

```
<type>(<scope>): <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `refactor`: Code refactoring
- `docs`: Documentation changes
- `test`: Test additions or modifications
- `perf`: Performance improvements
- `chore`: Maintenance tasks
- `ci`: CI/CD changes

### Pull Request Process

1. Create a feature branch from `main`
2. Make your changes following our coding standards
3. Write or update tests as needed
4. Ensure all tests pass: `npm test`
5. Run linting: `npm run lint`
6. Update documentation if necessary
7. Submit a pull request with a clear description

### Code Review Criteria

- Code follows TypeScript best practices
- Adequate test coverage (minimum 80%)
- No security vulnerabilities
- Performance considerations addressed
- Documentation updated
- CI/CD pipeline passes

## Coding Standards

### TypeScript Guidelines

- Use strict typing
- Avoid `any` type
- Implement proper error handling
- Use async/await over callbacks
- Follow functional programming principles where appropriate

### File Organization

```
src/
├── ai/           # AI and ML components
├── blockchain/   # Solana blockchain integration
├── config/       # Configuration management
├── controllers/  # API controllers
├── core/         # Core system components
├── middleware/   # Express middleware
├── services/     # Business logic services
├── test/         # Test utilities and fixtures
└── utils/        # Utility functions
```

### Testing Requirements

- Unit tests for all services and utilities
- Integration tests for API endpoints
- E2E tests for critical user flows
- Minimum 80% code coverage
- Use Jest for testing framework

### Documentation Standards

- JSDoc comments for all public functions
- README files for each major module
- API documentation using OpenAPI/Swagger
- Architecture decision records (ADRs) for significant changes

## Architecture Guidelines

### Design Principles

1. **Separation of Concerns**: Clear boundaries between layers
2. **Single Responsibility**: Each module has one clear purpose
3. **Dependency Injection**: Use DI for better testability
4. **Event-Driven Architecture**: Leverage event emitters for decoupling
5. **Immutability**: Prefer immutable data structures

### Performance Considerations

- Implement caching strategies
- Use connection pooling
- Optimize database queries
- Implement rate limiting
- Use WebSocket for real-time features
- Batch processing for heavy operations

### Security Requirements

- Input validation on all endpoints
- SQL injection prevention
- XSS protection
- Rate limiting implementation
- Authentication and authorization
- Secure secret management
- Regular dependency updates

## Testing

### Running Tests

```bash
# Run all tests
npm test

# Run tests in watch mode
npm run test:watch

# Run integration tests
npm run test:integration

# Run E2E tests
npm run test:e2e

# Generate coverage report
npm run test:coverage
```

### Writing Tests

- Use descriptive test names
- Follow AAA pattern (Arrange, Act, Assert)
- Mock external dependencies
- Test edge cases
- Ensure tests are deterministic

## Deployment

### Docker Deployment

```bash
# Build Docker image
npm run docker:build

# Run with Docker Compose
npm run docker:run
```

### Production Considerations

- Environment variable validation
- Health check endpoints
- Graceful shutdown handling
- Log aggregation setup
- Monitoring and alerting
- Database backup strategy

## Release Process

1. Update version in package.json
2. Update CHANGELOG.md
3. Create release branch
4. Run full test suite
5. Build and test Docker image
6. Create pull request to main
7. After merge, tag release
8. Deploy to production

## Getting Help

- Check existing issues and pull requests
- Review documentation in `/docs`
- Contact maintainers via issues
- Join our Discord community

## License

By contributing to NeuroPulse, you agree that your contributions will be licensed under the MIT License.
